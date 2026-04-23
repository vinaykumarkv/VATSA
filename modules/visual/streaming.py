import cv2
import torch
import time
import numpy as np
from PIL import Image
from collections import deque


class VATSA_VideoStream:
    """
    Real-time streaming detection for VATSA V-Module.
    Supports webcam and video file input.
    """

    def __init__(self, pipeline, source=0, target_fps=15):
        self.pipeline = pipeline
        self.source = source
        self.target_fps = target_fps

        self.fps_history = deque(maxlen=30)
        self.detection_counts = deque(maxlen=30)
        self.embedding_times = deque(maxlen=30)

        # Consistent colours per label
        self.label_colors = {}
        self._color_palette = [
            (255, 80, 80), (80, 255, 80), (80, 80, 255),
            (255, 80, 255), (80, 255, 255), (255, 200, 80),
            (200, 80, 255), (80, 200, 255),
        ]

    def _get_label_color(self, label):
        if label not in self.label_colors:
            idx = len(self.label_colors) % len(self._color_palette)
            self.label_colors[label] = self._color_palette[idx]
        return self.label_colors[label]

    def run(self, show=True, save_path=None):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {self.source}")

        writer = None
        if save_path:
            fps = cap.get(cv2.CAP_PROP_FPS) or 15
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )

        print("Streaming started — press Q to quit")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Stream ended.")
                    break

                t_start = time.perf_counter()
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                t_embed_start = time.perf_counter()
                results = self.pipeline.process_image(pil_image)
                embed_ms = (time.perf_counter() - t_embed_start) * 1000

                self.embedding_times.append(embed_ms)
                self.detection_counts.append(len(results))

                # Draw bounding boxes on frame
                annotated_frame = self._draw_boxes(frame, results)

                # Draw HUD overlays
                fps = 1.0 / (time.perf_counter() - t_start + 1e-9)
                self.fps_history.append(fps)
                annotated_frame = self._draw_hud(annotated_frame, fps, embed_ms, results)

                if show:
                    cv2.imshow("VATSA V-Module", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Stopped by user.")
                        break

                if writer:
                    writer.write(annotated_frame)

        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

        return self.get_benchmark()

    def _draw_boxes(self, frame, results):
        """Draw bounding boxes with label + confidence + embedding preview."""
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            color = self._get_label_color(r["label"])
            conf = r["confidence"]

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Confidence bar inside top of box
            bar_width = x2 - x1
            filled = int(bar_width * conf)
            cv2.rectangle(frame, (x1, y1), (x1 + filled, y1 + 5), color, -1)

            # Label chip above box
            label_text = f"{r['label'].upper()}  {conf*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            chip_y = max(y1 - th - 10, 0)
            cv2.rectangle(frame, (x1, chip_y), (x1 + tw + 10, chip_y + th + 8), color, -1)
            cv2.putText(frame, label_text, (x1 + 5, chip_y + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            # Embedding preview below box (first 3 dims)
            emb = r["embedding"].cpu().tolist()
            emb_text = f"[{emb[0]:.2f}, {emb[1]:.2f}, {emb[2]:.2f}...]"
            cv2.putText(frame, emb_text, (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

        return frame

    def _draw_hud(self, frame, fps, embed_ms, results):
        """Top-left system stats + right-side detection panel."""
        h, w = frame.shape[:2]
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        avg_embed = sum(self.embedding_times) / len(self.embedding_times) if self.embedding_times else 0

        # ── Top-left stats panel ──────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (280, 120), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, "VATSA  V-MODULE", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 120), 2)

        fps_color = (0, 255, 0) if avg_fps > 10 else (0, 165, 255) if avg_fps > 5 else (0, 0, 255)
        cv2.putText(frame, f"FPS   {fps:.1f}  avg {avg_fps:.1f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, fps_color, 1)
        cv2.putText(frame, f"EMBED {embed_ms:.1f}ms  avg {avg_embed:.1f}ms", (10, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
        cv2.putText(frame, f"DETECTIONS  {len(results)}", (10, 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 220, 0), 1)

        # ── Right-side detection list panel ───────────────────────────
        if results:
            panel_w = 260
            panel_h = min(len(results) * 52 + 36, h)
            panel_x = w - panel_w - 10
            panel_y = 10

            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (panel_x, panel_y),
                          (panel_x + panel_w, panel_y + panel_h), (10, 10, 10), -1)
            cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)

            cv2.putText(frame, "DETECTED OBJECTS", (panel_x + 8, panel_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

            for i, r in enumerate(results):
                row_y = panel_y + 36 + i * 52
                if row_y + 50 > h:
                    break

                color = self._get_label_color(r["label"])
                conf = r["confidence"]

                # Label + confidence %
                cv2.putText(frame, f"{r['label'].upper()}", (panel_x + 8, row_y + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.putText(frame, f"{conf*100:.1f}%", (panel_x + 160, row_y + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                # Confidence bar
                bar_x, bar_y = panel_x + 8, row_y + 22
                bar_total = panel_w - 20
                bar_filled = int(bar_total * conf)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_total, bar_y + 8), (60, 60, 60), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_filled, bar_y + 8), color, -1)

                # Embedding norm (single meaningful scalar)
                emb_tensor = r["embedding"]
                emb_norm = float(torch.norm(emb_tensor).item())
                cv2.putText(frame, f"emb norm: {emb_norm:.2f}", (panel_x + 8, row_y + 46),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 140, 140), 1)

        return frame

    def get_benchmark(self):
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        avg_embed = sum(self.embedding_times) / len(self.embedding_times) if self.embedding_times else 0
        avg_det = sum(self.detection_counts) / len(self.detection_counts) if self.detection_counts else 0

        report = {
            "avg_fps": round(avg_fps, 2),
            "avg_embed_ms": round(avg_embed, 2),
            "avg_detections_per_frame": round(avg_det, 2),
            "total_frames": len(self.fps_history),
        }

        print("\n── VATSA V-Module Benchmark ──────────────────")
        for k, v in report.items():
            print(f"  {k:<30} {v}")
        print("──────────────────────────────────────────────")
        return report