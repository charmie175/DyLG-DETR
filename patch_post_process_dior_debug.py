#!/usr/bin/env python3
from pathlib import Path
import shutil

path = Path("/root/DyLG-DETR/ppdet/modeling/post_process.py")
backup = path.with_suffix(".py.bak_before_dior_debug")

if not path.exists():
    raise FileNotFoundError(f"Not found: {path}")

text = path.read_text()

old = """        else:
            scores, index = paddle.topk(
                scores.flatten(1), self.num_top_queries, axis=-1)
            labels = index % self.num_classes
            index = index // self.num_classes
            batch_ind = paddle.arange(end=scores.shape[0]).unsqueeze(-1).tile(
                [1, self.num_top_queries])
            index = paddle.stack([batch_ind, index], axis=-1)
            bbox_pred = paddle.gather_nd(bbox_pred, index)
"""

new = """        else:
            # ---- DIOR / RT-DETR post-process runtime diagnostics ----
            print('\\\\n========== DETR POSTPROCESS DEBUG ==========', flush=True)
            print('bbox_pred.shape =', bbox_pred.shape, flush=True)
            print('logits.shape =', logits.shape, flush=True)
            print('scores.shape(before flatten) =', scores.shape, flush=True)
            print('self.num_classes =', self.num_classes, flush=True)
            print('self.num_top_queries =', self.num_top_queries, flush=True)
            print('use_focal_loss =', self.use_focal_loss, flush=True)

            # Catch class-dimension mismatch before CUDA gather.
            assert logits.shape[-1] == self.num_classes, (
                'CLASS MISMATCH: '
                f'logits.shape={logits.shape}, '
                f'logits_classes={logits.shape[-1]}, '
                f'postprocess_num_classes={self.num_classes}'
            )

            # bbox_pred and logits should have the same query dimension.
            assert bbox_pred.shape[1] == logits.shape[1], (
                'QUERY DIMENSION MISMATCH: '
                f'bbox_pred.shape={bbox_pred.shape}, '
                f'logits.shape={logits.shape}'
            )

            flat_scores = scores.flatten(1)

            print('flat_scores.shape =', flat_scores.shape, flush=True)

            # topk itself also requires k <= flattened score dimension.
            assert self.num_top_queries <= flat_scores.shape[1], (
                'TOPK OUT OF RANGE: '
                f'num_top_queries={self.num_top_queries}, '
                f'flat_scores_dim={flat_scores.shape[1]}, '
                f'scores.shape={scores.shape}'
            )

            scores, flat_index = paddle.topk(
                flat_scores, self.num_top_queries, axis=-1)

            labels = flat_index % self.num_classes
            query_index = flat_index // self.num_classes

            # Synchronize here so any earlier CUDA error is reported at the
            # real operation instead of later at paddle.to_tensor().
            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.synchronize()

            query_min = int(paddle.min(query_index).numpy())
            query_max = int(paddle.max(query_index).numpy())

            print('flat_index.shape =', flat_index.shape, flush=True)
            print('query_index.shape =', query_index.shape, flush=True)
            print('query index min =', query_min, flush=True)
            print('query index max =', query_max, flush=True)
            print('bbox query count =', bbox_pred.shape[1], flush=True)
            print('============================================\\\\n', flush=True)

            assert query_min >= 0, (
                f'NEGATIVE QUERY INDEX: min={query_min}'
            )

            assert query_max < bbox_pred.shape[1], (
                'QUERY INDEX OUT OF RANGE: '
                f'query_index_max={query_max}, '
                f'bbox_queries={bbox_pred.shape[1]}, '
                f'bbox_pred.shape={bbox_pred.shape}, '
                f'logits.shape={logits.shape}, '
                f'num_classes={self.num_classes}, '
                f'num_top_queries={self.num_top_queries}'
            )

            batch_ind = paddle.arange(
                end=scores.shape[0]).unsqueeze(-1).tile(
                    [1, self.num_top_queries])

            index = paddle.stack([batch_ind, query_index], axis=-1)

            print('gather index.shape =', index.shape, flush=True)
            print('starting bbox gather_nd ...', flush=True)

            bbox_pred = paddle.gather_nd(bbox_pred, index)

            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.synchronize()

            print('bbox gather_nd OK', flush=True)
"""

if old not in text:
    raise RuntimeError(
        "Target block was not found exactly. "
        "Your post_process.py differs from the snippet."
    )

if not backup.exists():
    shutil.copy2(path, backup)
    print(f"Backup created: {backup}")
else:
    print(f"Backup already exists: {backup}")

patched = text.replace(old, new, 1)
path.write_text(patched)

print(f"Patched successfully: {path}")
print("Run this to inspect the modified block:")
print("nl -ba /root/DyLG-DETR/ppdet/modeling/post_process.py | sed -n '100,190p'")
