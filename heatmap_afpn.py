import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, cv2, os, shutil
import numpy as np
np.random.seed(0)
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import DetectionModel as Model
from ultralytics.utils.torch_utils import intersect_dicts
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


class yolov8_heatmap:
    def __init__(self, weight, cfg, device, method, layer, conf_threshold):
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names
        csd = ckpt['model'].float().state_dict()
        model = Model(cfg, ch=3, nc=len(model_names)).to(device)
        csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])
        model.load_state_dict(csd, strict=False)
        model.eval()
        print(f'Transferred {len(csd)}/{len(model.state_dict())} items')

        self.device = device
        self.model = model
        self.model_names = model_names
        self.nc = len(model_names)
        self.method = eval(method)
        self.layer = layer  # string like "model.model[10].afpn.body[0]"
        self.conf_threshold = conf_threshold
        self.colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(int)

    def post_process(self, result):
        """
        result: [4+nc, N] — decoded output from Detect.forward()
          channels 0~3:  xywh boxes in pixel coords (already decoded)
          channels 4~:   sigmoid-ed class confidences
        """
        nc = self.nc
        boxes = result[:4, :]      # [4, N] decoded xywh, pixel coords
        scores = result[4:, :]     # [nc, N] sigmoid-ed

        max_scores = scores.max(dim=0)[0]  # [N]
        top_val, top_idx = max_scores.max(dim=0)

        best_box = boxes[:, top_idx]       # [4] xywh
        best_scores = scores[:, top_idx]   # [nc]

        x_ctr, y_ctr, w, h = best_box[0], best_box[1], best_box[2], best_box[3]
        x1 = x_ctr - w / 2
        y1 = y_ctr - h / 2
        x2 = x_ctr + w / 2
        y2 = y_ctr + h / 2
        xyxy = torch.stack([x1, y1, x2, y2]).cpu().detach().numpy()

        return best_scores, best_box, xyxy, int(top_idx.item())

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    tuple(int(x) for x in color), 2, lineType=cv2.LINE_AA)
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

        # --- 1st forward (no_grad): find which scale the top detection is on ---
        with torch.no_grad():
            result = self.model(tensor)
            if isinstance(result, (tuple, list)):
                result = result[0]
            _, _, _, top_idx = self.post_process(result[0])

        # Determine which FPN scale top_idx belongs to
        detect = self.model.model[-1]  # Detect_AFPN_P2345_Custom
        strides = detect.stride.cpu().numpy()  # e.g. [4, 8, 16, 32] for P2-P5
        anchors_per_scale = [int((640 / s) ** 2) for s in strides]
        cumsum = np.cumsum([0] + anchors_per_scale)
        scale_idx = 0
        for s in range(len(strides)):
            if cumsum[s] <= top_idx < cumsum[s + 1]:
                scale_idx = s
                break
        print(f'Detection on P{2+scale_idx}/{int(strides[scale_idx])} (scale {scale_idx}) '
              f'→ anchor {top_idx} of {cumsum[-1]}')

        # Dynamically select AFPN output conv based on detection scale
        scale_to_conv = {0: 'conv00', 1: 'conv11', 2: 'conv22', 3: 'conv33'}
        conv_name = scale_to_conv[scale_idx]
        target_layer = getattr(self.model.model[10].afpn, conv_name)
        layer_desc = f'self.model.model[10].afpn.{conv_name}'
        print(f'Target layer: {layer_desc} (P{2+scale_idx}/{int(strides[scale_idx])}, '
              f'{int(640/strides[scale_idx])}×{int(640/strides[scale_idx])})')

        # --- 2nd forward (with hooks) ---
        grads = ActivationsAndGradients(self.model, [target_layer], reshape_transform=None)
        result = grads(tensor)
        if isinstance(result, (tuple, list)):
            result = result[0]
        y_decoded = result[0]
        activations = grads.activations[0].cpu().detach().numpy()

        # --- extract top-1 detection ---
        post_result, pre_post_boxes, post_boxes, _ = self.post_process(y_decoded)
        conf = float(post_result.max())
        cls_name = self.model_names[int(post_result.argmax())]
        print(f'Top-1: {cls_name} conf={conf:.4f} box={post_boxes}')

        for i in trange(1):
            if conf < self.conf_threshold:
                print(f'Skipped: confidence {conf:.4f} < threshold {self.conf_threshold}')
                break

            self.model.zero_grad()
            score = post_result.max()
            score.backward(retain_graph=True)

            if len(grads.gradients) == 0:
                print('ERROR: grads.gradients is empty! Try a different target layer.')
                break
            gradients = grads.gradients[0]
            print(f'Gradient shape: {gradients.shape}, nonzero: {(gradients != 0).sum().item()}')

            b, k, u, v = gradients.size()
            act_np = activations
            grad_np = gradients.detach().cpu().numpy()
            weights = self.method.get_cam_weights(self.method, None, None, None, act_np, grad_np)
            weights = weights.reshape((b, k, 1, 1))
            saliency_map = np.sum(weights * act_np, axis=1)
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
                                             f'{cls_name} {conf:.2f}', cam_image)
            cam_image = Image.fromarray(cam_image)
            cam_image.save(f'{save_path}/{i}.png')
            print(f'Heatmap saved to {save_path}/{i}.png')


def get_params():
    return {
        'weight': 'runs/train/SLM-AFPN-C2fSCConv/weights/best.pt',
        'cfg': 'ultralytics/cfg/models/v8/yolov8-AFPN-P2345-Custom.yaml',
        'device': 'cuda:0',
        'method': 'GradCAMPlusPlus',
        # Target layer is dynamically selected based on which FPN scale
        # the detection falls on (P2→conv00, P3→conv11, P4→conv22, P5→conv33).
        # The 'layer' param below is no longer used; kept for reference.
        'layer': "dynamic (auto-select AFPN output conv by detection scale)",
        'conf_threshold': 0.1,
    }


if __name__ == '__main__':
    model = yolov8_heatmap(**get_params())
    model(r'D:/app/datasettag/test/images/WIN_20250821_01_48_25_Pro.jpg', 'result_afpn')
