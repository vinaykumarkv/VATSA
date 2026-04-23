import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms


class VATSA_Detector:
    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.3):
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold

        self.crop_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])

    def detect_and_crop(self, pil_image):
        results = self.model(pil_image, conf=self.conf_threshold)[0]
        regions = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            label = results.names[int(box.cls[0])]
            crop = pil_image.crop((x1, y1, x2, y2))
            crop_tensor = self.crop_transform(crop)

            regions.append({
                "label": label,
                "confidence": conf,
                "crop_tensor": crop_tensor,
                "bbox": (x1, y1, x2, y2)
            })

        return regions

    def visualise(self, pil_image, results, save_path=None):
        """
        Draws bounding boxes + labels + embeddings preview on image.
        results = output from VATSA_VisualPipeline.process_image()
        """
        img = pil_image.copy()
        draw = ImageDraw.Draw(img)

        colors = ["#FF4444", "#44FF44", "#4444FF", "#FF44FF", "#44FFFF", "#FFFF44"]

        for i, r in enumerate(results):
            color = colors[i % len(colors)]
            x1, y1, x2, y2 = r["bbox"]

            # Bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Label background
            label_text = f"{r['label']} {r['confidence']:.2f}"
            text_bbox = draw.textbbox((x1, y1), label_text)
            draw.rectangle([text_bbox[0]-2, text_bbox[1]-2,
                            text_bbox[2]+2, text_bbox[3]+2], fill=color)
            draw.text((x1, y1), label_text, fill="white")

            # Embedding preview (first 4 values)
            emb = r["embedding"].cpu().tolist()
            emb_text = f"emb: [{emb[0]:.2f}, {emb[1]:.2f}, {emb[2]:.2f}...]"
            draw.text((x1, y2 + 4), emb_text, fill=color)

        if save_path:
            img.save(save_path)
            print(f"Saved to {save_path}")

        return img