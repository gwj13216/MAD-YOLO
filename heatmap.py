import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import DetectionModel as Model
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

class yolov8_heatmap:
    def __init__(self, weight, cfg, device, method, layer, backward_type, conf_threshold, ratio):
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        model = Model(cfg, ch=3, nc=len(model_names)).to(device)
        csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])  # intersect
        model.load_state_dict(csd, strict=False)  # load
        model.eval()
        print(f'Transferred {len(csd)}/{len(model.state_dict())} items')
        
        target_layers = [eval(layer)]
        method = eval(method)

        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(int)
        self.__dict__.update(locals())
    
    def post_process(self, result):
        """
        result: tensor of shape [C, N] where C = 4 + num_classes (batch dim removed).
        YOLOv8 Detect.forward() already decodes boxes (DFL→dist2bbox→×strides) and
        applies sigmoid to class scores. Output format: [4+nc, num_anchors]
          - channels 0~3:  decoded xywh boxes in pixel coordinates (NOT 0-1)
          - channels 4~:   sigmoid-ed class confidences (NOT raw logits)
        Returns: class_scores [nc], raw_boxes [4], decoded_xyxy [4]
        """
        nc = len(self.model_names)
        boxes = result[:4, :]      # [4, N] — decoded xywh, pixel coords
        scores = result[4:, :]     # [nc, N] — already sigmoid-ed

        # Find the top-1 detection across all anchors
        max_scores = scores.max(dim=0)[0]     # [N]
        top_val, top_idx = max_scores.max(dim=0)  # best anchor (scalar)

        best_box = boxes[:, top_idx]           # [4] — xywh of top prediction
        best_scores = scores[:, top_idx]       # [nc] — class scores of top prediction

        # Convert xywh(center) → xyxy(corner), boxes are already in pixel coords
        x_ctr, y_ctr, w, h = best_box[0], best_box[1], best_box[2], best_box[3]
        x1 = x_ctr - w / 2
        y1 = y_ctr - h / 2
        x2 = x_ctr + w / 2
        y2 = y_ctr + h / 2
        xyxy = torch.stack([x1, y1, x2, y2]).cpu().detach().numpy()

        return best_scores, best_box, xyxy
    
    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2, lineType=cv2.LINE_AA)
        return img

    def __call__(self, img_path, save_path):
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

        # --- load & preprocess ---
        img_orig = cv2.imread(img_path)
        img = letterbox(img_orig)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        # --- 1st forward (no_grad): find which FPN scale the top detection is on ---
        with torch.no_grad():
            result = self.model(tensor)
            if isinstance(result, (tuple, list)):
                result = result[0]
            _, _, top_idx = self._find_top_detection(result[0])

        # Map anchor → FPN scale → target layer
        detect = self.model.model[-1]
        strides = detect.stride.cpu().numpy()  # [8, 16, 32]
        anchors_per_scale = [int((640 / s) ** 2) for s in strides]
        cumsum = np.cumsum([0] + anchors_per_scale)
        scale_idx = 0
        for s in range(3):
            if cumsum[s] <= top_idx < cumsum[s + 1]:
                scale_idx = s
                break
        scale_to_layer = {0: 15, 1: 18, 2: 21}
        target_layer = self.model.model[scale_to_layer[scale_idx]]
        print(f'Detection on P{3+scale_idx}/{int(strides[scale_idx])} → target layer model.model[{scale_to_layer[scale_idx]}]')

        # --- 2nd forward (with hook): GradCAM on the CORRECT layer ---
        grads = ActivationsAndGradients(self.model, [target_layer], reshape_transform=None)
        result = grads(tensor)
        if isinstance(result, (tuple, list)):
            result = result[0]
        y_decoded = result[0]
        activations = grads.activations[0].cpu().detach().numpy()

        # --- extract top-1 detection ---
        post_result, pre_post_boxes, post_boxes = self.post_process(y_decoded)
        print(f'Top-1 confidence: {float(post_result.max()):.4f}, threshold: {self.conf_threshold}')
        print(f'Top-1 class: {self.model_names[int(post_result.argmax())]}')
        print(f'Top-1 box: {post_boxes}')

        for i in trange(1):
            if float(post_result.max()) < self.conf_threshold:
                print(f'Skipped: confidence {float(post_result.max()):.4f} < threshold {self.conf_threshold}')
                break

            self.model.zero_grad()
            score = post_result.max()
            score.backward(retain_graph=True)

            if len(grads.gradients) == 0:
                print('ERROR: grads.gradients is empty!')
                break
            gradients = grads.gradients[0]
            print(f'Gradient shape: {gradients.shape}, nonzero: {(gradients != 0).sum().item()}')

            b, k, u, v = gradients.size()
            weights = self.method.get_cam_weights(self.method, None, None, None, activations, gradients.detach().numpy())
            weights = weights.reshape((b, k, 1, 1))
            saliency_map = np.sum(weights * activations, axis=1)
            saliency_map = np.squeeze(np.maximum(saliency_map, 0))
            saliency_map = cv2.resize(saliency_map, (tensor.size(3), tensor.size(2)))
            smin, smax = saliency_map.min(), saliency_map.max()
            print(f'Saliency range: [{smin:.6f}, {smax:.6f}]')
            if smax - smin == 0:
                print('WARNING: saliency map is uniform — skipping save')
                continue

            saliency_map = (saliency_map - smin) / (smax - smin)
            cam_image = show_cam_on_image(img.copy(), saliency_map, use_rgb=True)
            cam_image = self.draw_detections(post_boxes, self.colors[int(post_result.argmax())],
                                             f'{self.model_names[int(post_result.argmax())]} {float(post_result.max()):.2f}', cam_image)
            cam_image = Image.fromarray(cam_image)
            cam_image.save(f'{save_path}/{i}.png')
            print(f'Heatmap saved to {save_path}/{i}.png')

    def _find_top_detection(self, y_decoded):
        """Return (best_scores, best_box_xywh, anchor_index) for the top-1 detection."""
        boxes = y_decoded[:4, :]
        scores = y_decoded[4:, :]
        max_scores = scores.max(dim=0)[0]
        top_val, top_idx = max_scores.max(dim=0)
        return scores[:, top_idx], boxes[:, top_idx], int(top_idx.item())

def get_params():
    params = {
        'weight': 'runs/train/yolov8/weights/best.pt',
        'cfg': 'ultralytics/cfg/models/v8/yolov8.yaml',
        'device': 'cuda:0',
        'method': 'GradCAMPlusPlus', # GradCAMPlusPlus, GradCAM, XGradCAM
        # target layer for GradCAM — choose based on object size:
        #   model.model[15]  P3/8  C2f  80×80  (best for small objects)
        #   model.model[18]  P4/16 C2f  40×40  (balanced, recommended)
        #   model.model[21]  P5/32 C2f  20×20  (best for large objects)
        'layer': 'model.model[15]',  # P3/8 C2f 80×80 — use this for small objects (mosquito)
        'backward_type': 'class', # class, box, all
        'conf_threshold': 0.1, # 0.6
        'ratio': 0.02 # 0.02-0.1
    }
    return params

if __name__ == '__main__':
    model = yolov8_heatmap(**get_params())
    model(r'D:/app/datasettag/test/images/WIN_20250821_01_48_25_Pro.jpg', 'result')