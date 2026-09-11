"""
Batch Grad-CAM++ heatmap generation for AFPN model (SLM-AFPN-C2fSCConv).
Processes all images in the test set.
"""
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, cv2, os, shutil, glob
import numpy as np
np.random.seed(0)
from tqdm import tqdm
from PIL import Image
from ultralytics.nn.tasks import DetectionModel as Model
from ultralytics.utils.torch_utils import intersect_dicts
from pytorch_grad_cam import GradCAMPlusPlus
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


def post_process(result, nc):
    boxes = result[:4, :]
    scores = result[4:, :]
    max_scores = scores.max(dim=0)[0]
    top_val, top_idx = max_scores.max(dim=0)
    best_box = boxes[:, top_idx]
    best_scores = scores[:, top_idx]
    x_ctr, y_ctr, w, h = best_box[0], best_box[1], best_box[2], best_box[3]
    xyxy = torch.stack([x_ctr - w / 2, y_ctr - h / 2, x_ctr + w / 2, y_ctr + h / 2]).cpu().detach().numpy()
    return best_scores, best_box, xyxy, int(top_idx.item())


def draw_detections(img, box, color, name):
    xmin, ymin, xmax, ymax = list(map(int, list(box)))
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
    cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                tuple(int(x) for x in color), 2, lineType=cv2.LINE_AA)
    return img


def main():
    weight = 'runs/train/SLM-AFPN-C2fSCConv/weights/best.pt'
    cfg = 'ultralytics/cfg/models/v8/yolov8-AFPN-P2345-Custom.yaml'
    device = torch.device('cuda:0')
    conf_threshold = 0.1
    test_dir = 'D:/app/datasettag/test/images'
    save_root = 'heatmap_results/afpn'

    # --- Load model ---
    ckpt = torch.load(weight)
    model_names = ckpt['model'].names
    nc = len(model_names)
    csd = ckpt['model'].float().state_dict()
    model = Model(cfg, ch=3, nc=nc).to(device)
    csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])
    model.load_state_dict(csd, strict=False)
    model.eval()
    print(f'Model loaded: {len(csd)}/{len(model.state_dict())} items, {nc} classes')

    method = GradCAMPlusPlus
    colors = np.random.uniform(0, 255, size=(nc, 3)).astype(int)

    # --- Detect head info (AFPN: 4 scales P2-P5) ---
    detect = model.model[-1]
    strides = detect.stride.cpu().numpy()
    anchors_per_scale = [int((640 / s) ** 2) for s in strides]
    cumsum = np.cumsum([0] + anchors_per_scale)
    scale_to_conv = {0: 'conv00', 1: 'conv11', 2: 'conv22', 3: 'conv33'}

    # --- Get image list ---
    img_paths = sorted(glob.glob(os.path.join(test_dir, '*.jpg')))
    print(f'Found {len(img_paths)} images')

    success, skipped_low_conf, skipped_uniform = 0, 0, 0

    for img_path in tqdm(img_paths, desc='Generating heatmaps'):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        save_dir = os.path.join(save_root, img_name)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        # --- Preprocess ---
        img_orig = cv2.imread(img_path)
        img = letterbox(img_orig)[0]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = np.float32(img_rgb) / 255.0
        tensor = torch.from_numpy(np.transpose(img_norm, axes=[2, 0, 1])).unsqueeze(0).to(device)

        # --- 1st forward: find scale ---
        with torch.no_grad():
            result = model(tensor)
            if isinstance(result, (tuple, list)):
                result = result[0]
            _, _, _, top_idx = post_process(result[0], nc)

        scale_idx = 0
        for s in range(len(strides)):
            if cumsum[s] <= top_idx < cumsum[s + 1]:
                scale_idx = s
                break
        target_layer = getattr(model.model[-1].afpn, scale_to_conv[scale_idx])

        # --- 2nd forward: GradCAM on correct layer ---
        grads = ActivationsAndGradients(model, [target_layer], reshape_transform=None)
        result = grads(tensor)
        if isinstance(result, (tuple, list)):
            result = result[0]
        y_decoded = result[0]
        activations = grads.activations[0].cpu().detach().numpy()

        post_result, pre_post_boxes, post_boxes, _ = post_process(y_decoded, nc)
        conf = float(post_result.max())
        cls_name = model_names[int(post_result.argmax())]

        if conf < conf_threshold:
            skipped_low_conf += 1
            continue

        model.zero_grad()
        score = post_result.max()
        score.backward(retain_graph=True)

        if len(grads.gradients) == 0:
            skipped_uniform += 1
            continue

        gradients = grads.gradients[0]
        b, k, u, v = gradients.size()
        act_np = activations
        grad_np = gradients.detach().cpu().numpy()
        weights = method.get_cam_weights(method, None, None, None, act_np, grad_np)
        weights = weights.reshape((b, k, 1, 1))
        saliency_map = np.sum(weights * act_np, axis=1)
        saliency_map = np.squeeze(np.maximum(saliency_map, 0))
        saliency_map = cv2.resize(saliency_map, (tensor.size(3), tensor.size(2)))
        smin, smax = saliency_map.min(), saliency_map.max()
        if smax - smin == 0:
            skipped_uniform += 1
            continue

        saliency_map = (saliency_map - smin) / (smax - smin)
        cam_image = show_cam_on_image(img_rgb.copy() / 255.0, saliency_map, use_rgb=True)
        cam_image = draw_detections(cam_image, post_boxes, colors[int(post_result.argmax())],
                                    f'{cls_name} {conf:.2f}')
        Image.fromarray(cam_image).save(f'{save_dir}/heatmap.png')
        success += 1

    print(f'\nDone. Success: {success}, Low confidence: {skipped_low_conf}, '
          f'Uniform/empty: {skipped_uniform}, Total: {len(img_paths)}')
    print(f'Results saved to: {save_root}')


if __name__ == '__main__':
    main()
