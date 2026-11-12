<<<<<<< HEAD
# DyLG-DETR

## Caveat

- DyLG-DETR uses 4 GPUs for training.

## DIOR per-class AP

The downloaded DIOR archive already contains COCO JSON under
`DIOR/Annotations/{train,val,test}.json`; copy/extract those files to the
dataset `annotations/` directory and use `val.json` for evaluation.
For an alternate VOC-style DIOR copy, convert annotations to COCO JSON first:

```powershell
python tools/dior_voc2coco.py `
  --dior-root D:/data/DIOR `
  --split val `
  --output D:/data/DIOR/annotations/val.json
```

Then evaluate each model's COCO detection result:

```powershell
python tools/eval_per_class.py `
  --anno D:/data/DIOR/annotations/val.json `
  --pred output/dior_dylg/bbox.json `
  --method DyLG-DETR `
  --output experiments/dior_dylg_per_class.csv
```

The CSV reports AP averaged over IoU 0.50:0.95, AP50, and ground-truth count
for every DIOR category. Run the command once per method and merge the files
into `experiments/dior_comparison.csv`. The comparison template includes the
reviewer's requested RTMDet, PKINet, Strip R-CNN, and RHINO-DETR rows; fill
literature rows only with values from the cited paper or an independently
reproduced checkpoint, and keep the `source` field to distinguish them from
measurements on the same DIOR split.

To generate a wide table from measured files:

```powershell
python tools/merge_per_class.py `
  --input RT-DETR=experiments/rtdetr_per_class.csv `
  --input DyLG-DETR=experiments/dylg_per_class.csv `
  --output experiments/dior_comparison_measured.csv
```

The DIOR archive is expected at `D:/dataset/dior_extract/DIOR` after
extraction. The training configuration is
`configs/rtdetr/rtdetr_r50vd_6x_dior.yml`; it initializes from the local
COCO-pretrained RT-DETR checkpoint and trains the 20-category DIOR detector:

```powershell
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_dior.yml --eval
```

After training, evaluate the saved checkpoint with `tools/eval.py` and then
pass its generated `bbox.json` to `tools/eval_per_class.py`.

For the local RTX 5060, run the WSL2 configuration from an Ubuntu shell:

```bash
cd /mnt/d/DyLG-DETR
source /home/qlm/miniconda3/etc/profile.d/conda.sh
conda activate tvlight
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_dior_wsl.yml
```

## Running from VS Code (WSL2)

Install the **Remote - WSL** and **Python** extensions, then open this folder
in the Ubuntu-22.04 environment (`\\wsl$\\Ubuntu-22.04\\mnt\\d\\DyLG-DETR`,
or run `code /mnt/d/DyLG-DETR` from WSL). Select the interpreter
`/home/qlm/miniconda3/envs/tvlight/bin/python` when prompted.

The Run and Debug panel contains:

- `DIOR: train 72 epochs (RTX 5060)`: AMP, fixed 640 input, batch size 1,
  and reduced denoising/query counts for the 8 GB card. Output is written to
  `output/rtdetr_r50vd_6x_dior_wsl` and streams in the integrated terminal.
- `DIOR: evaluate final checkpoint`: evaluates that checkpoint with classwise
  COCO AP enabled.

The unmodified 600-query profile is retained as
`configs/rtdetr/rtdetr_r50vd_6x_dior_wsl.yml`; it exceeds 8 GB on the RTX 5060
even with AMP.


## Data and Materials Availability

The original research materials supporting the
DyLG-DETR study are made available under the
Creative Commons Attribution 4.0 International
(CC BY 4.0) License, where explicitly indicated.

License details: [LICENSE-DATA.md](LICENSE-DATA.md)

The benchmark datasets (DIOR, DOTA-v1.0,
SODA-A, and VisDrone2019-DET) are not covered
by this licence and remain subject to their
respective original licences.
=======
# DyLG-DETR
>>>>>>> 6f5e1ba3779277137d879ca5839e11996ccae1cf
