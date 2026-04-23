import torch
import time
import json
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict
from datetime import datetime


class VATSA_Benchmark:
    """
    Comprehensive benchmark for VATSA V-Module.
    Tests latency, throughput, embedding quality and saves a report.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.results = defaultdict(list)

    # ── 1. LATENCY TEST ───────────────────────────────────────────────
    def run_latency_test(self, image_paths=None, n_runs=50):
        """
        Measures per-image breakdown:
        preprocess → detection → embedding → total
        """
        print("\n── Latency Test ──────────────────────────────")

        # Use provided images or generate synthetic ones
        images = self._load_or_generate(image_paths, n=n_runs)

        detect_times, embed_times, total_times = [], [], []

        for i, img in enumerate(images):
            # Total pipeline
            t0 = time.perf_counter()

            # Detection only
            t1 = time.perf_counter()
            regions = self.pipeline.detector.detect_and_crop(img)
            t2 = time.perf_counter()

            # Embedding only (skip if no detections)
            if regions:
                import torch
                crops = torch.stack(
                    [r["crop_tensor"] for r in regions]
                ).to(self.pipeline.device)
                with torch.no_grad():
                    _ = self.pipeline.encoder(crops)
            t3 = time.perf_counter()

            detect_ms = (t2 - t1) * 1000
            embed_ms = (t3 - t2) * 1000
            total_ms = (t3 - t0) * 1000

            detect_times.append(detect_ms)
            embed_times.append(embed_ms)
            total_times.append(total_ms)

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n_runs}] total: {total_ms:.1f}ms  "
                      f"detect: {detect_ms:.1f}ms  embed: {embed_ms:.1f}ms")

        self.results["latency"] = {
            "detection_ms":  self._stats(detect_times),
            "embedding_ms":  self._stats(embed_times),
            "total_ms":      self._stats(total_times),
        }

        print(f"\n  Detection  — mean: {np.mean(detect_times):.1f}ms  "
              f"p95: {np.percentile(detect_times, 95):.1f}ms")
        print(f"  Embedding  — mean: {np.mean(embed_times):.1f}ms  "
              f"p95: {np.percentile(embed_times, 95):.1f}ms")
        print(f"  Total      — mean: {np.mean(total_times):.1f}ms  "
              f"p95: {np.percentile(total_times, 95):.1f}ms")
        print(f"  Implied FPS: {1000/np.mean(total_times):.1f}")

        return self

    # ── 2. THROUGHPUT TEST ────────────────────────────────────────────
    def run_throughput_test(self, batch_sizes=[1, 4, 8, 16, 32]):
        """
        Tests encoder throughput at different batch sizes.
        Tells you max embeddings/sec your GPU can handle.
        """
        print("\n── Throughput Test ───────────────────────────")
        throughput_results = {}

        for bs in batch_sizes:
            # Synthetic batch of crops
            dummy = torch.randn(bs, 3, 224, 224).to(self.pipeline.device)

            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    self.pipeline.encoder(dummy)

            # Measure
            times = []
            with torch.no_grad():
                for _ in range(20):
                    t0 = time.perf_counter()
                    self.pipeline.encoder(dummy)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - t0) * 1000)

            mean_ms = np.mean(times)
            embeddings_per_sec = (bs / mean_ms) * 1000
            throughput_results[bs] = {
                "mean_ms": round(mean_ms, 2),
                "embeddings_per_sec": round(embeddings_per_sec, 1)
            }
            print(f"  Batch {bs:>2} — {mean_ms:.1f}ms — "
                  f"{embeddings_per_sec:.0f} embeddings/sec")

        self.results["throughput"] = throughput_results
        return self

    # ── 3. EMBEDDING QUALITY TEST ─────────────────────────────────────
    def run_embedding_quality_test(self, image_paths=None):
        """
        Tests embedding consistency:
        - Same image twice → embeddings should be identical
        - Augmented image → embeddings should be close (not identical)
        - Different images → embeddings should be far apart
        """
        import torchvision.transforms as transforms
        print("\n── Embedding Quality Test ────────────────────")

        images = self._load_or_generate(image_paths, n=10)

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])

        aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])

        consistency_scores = []
        augmentation_scores = []
        separation_scores = []

        embeddings = []
        with torch.no_grad():
            for img in images:
                t1 = transform(img).unsqueeze(0).to(self.pipeline.device)
                t2 = transform(img).unsqueeze(0).to(self.pipeline.device)
                aug = aug_transform(img).unsqueeze(0).to(self.pipeline.device)

                e1 = self.pipeline.encoder(t1)["embedding"]
                e2 = self.pipeline.encoder(t2)["embedding"]
                e_aug = self.pipeline.encoder(aug)["embedding"]

                # Cosine similarity
                cos = torch.nn.functional.cosine_similarity
                consistency_scores.append(cos(e1, e2).item())
                augmentation_scores.append(cos(e1, e_aug).item())
                embeddings.append(e1.squeeze())

        # Cross-image separation
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                cos_sim = torch.nn.functional.cosine_similarity(
                    embeddings[i].unsqueeze(0),
                    embeddings[j].unsqueeze(0)
                ).item()
                separation_scores.append(cos_sim)

        mean_consistency = np.mean(consistency_scores)
        mean_augmentation = np.mean(augmentation_scores)
        mean_separation = np.mean(separation_scores)

        self.results["embedding_quality"] = {
            "consistency":          round(mean_consistency, 4),
            "augmentation_robustness": round(mean_augmentation, 4),
            "inter_image_separation":  round(mean_separation, 4),
        }

        print(f"  Consistency (same→same):     {mean_consistency:.4f}  "
              f"{'✅ perfect' if mean_consistency > 0.999 else '⚠️ check'}")
        print(f"  Augmentation robustness:     {mean_augmentation:.4f}  "
              f"{'✅ good' if mean_augmentation > 0.8 else '⚠️ low'}")
        print(f"  Inter-image separation:      {mean_separation:.4f}  "
              f"{'✅ good' if mean_separation < 0.9 else '⚠️ embeddings too similar'}")

        return self

    # ── 4. GPU MEMORY TEST ────────────────────────────────────────────
    def run_memory_test(self):
        """Reports GPU memory usage of the pipeline."""
        print("\n── GPU Memory Test ───────────────────────────")

        if not torch.cuda.is_available():
            print("  CUDA not available — skipping")
            return self

        torch.cuda.reset_peak_memory_stats()
        dummy = torch.randn(1, 3, 224, 224).to(self.pipeline.device)

        with torch.no_grad():
            self.pipeline.encoder(dummy)

        allocated = torch.cuda.memory_allocated() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2

        self.results["gpu_memory"] = {
            "allocated_mb": round(allocated, 1),
            "peak_mb":      round(peak, 1),
            "reserved_mb":  round(reserved, 1),
        }

        print(f"  Allocated: {allocated:.1f} MB")
        print(f"  Peak:      {peak:.1f} MB")
        print(f"  Reserved:  {reserved:.1f} MB")

        return self

    # ── 5. SAVE REPORT ────────────────────────────────────────────────
    def save_report(self, path="vatsa_vmodule_benchmark.json"):
        report = {
            "module":    "VATSA V-Module",
            "timestamp": datetime.now().isoformat(),
            "device":    str(self.pipeline.device),
            "results":   dict(self.results)
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Benchmark report saved → {path}")
        return report

    def print_summary(self):
        print("\n══ VATSA V-Module — Benchmark Summary ═══════════════════")
        if "latency" in self.results:
            t = self.results["latency"]["total_ms"]
            print(f"  Pipeline latency   mean {t['mean']:.1f}ms  "
                  f"p95 {t['p95']:.1f}ms  "
                  f"→ {1000/t['mean']:.1f} FPS")
        if "throughput" in self.results:
            best = max(self.results["throughput"].items(),
                       key=lambda x: x[1]["embeddings_per_sec"])
            print(f"  Peak throughput    {best[1]['embeddings_per_sec']} emb/sec "
                  f"at batch size {best[0]}")
        if "embedding_quality" in self.results:
            q = self.results["embedding_quality"]
            print(f"  Embedding quality  consistency={q['consistency']}  "
                  f"aug_robustness={q['augmentation_robustness']}")
        if "gpu_memory" in self.results:
            m = self.results["gpu_memory"]
            print(f"  GPU memory         {m['allocated_mb']}MB allocated  "
                  f"{m['peak_mb']}MB peak")
        print("══════════════════════════════════════════════════════════")

    # ── HELPERS ───────────────────────────────────────────────────────
    def _load_or_generate(self, image_paths, n):
        if image_paths:
            return [Image.open(p).convert("RGB") for p in image_paths]
        # Synthetic RGB images if no real images provided
        return [Image.fromarray(
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        ) for _ in range(n)]

    def _stats(self, values):
        return {
            "mean": round(float(np.mean(values)), 2),
            "std":  round(float(np.std(values)), 2),
            "min":  round(float(np.min(values)), 2),
            "max":  round(float(np.max(values)), 2),
            "p95":  round(float(np.percentile(values, 95)), 2),
        }