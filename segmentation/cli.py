# cli.py

import click
from pathlib import Path
from .io import FileIndex, ImageIO
from .qc import plot_segmentation_slices
from .analysis import DistanceAnalyzer
from .logger import logger

@click.group()
def cli():
    pass

@cli.command()
@click.option('--raw-dir',      required=True,  type=Path, help="Folder of .nd2/.czi/.tif files")
@click.option('--denoised-dir', required=False, type=Path, help="Folder of denoised .tif files (optional)")
@click.option('--pattern',      default="*.nd2", show_default=True,
              help="Glob pattern for raw images (e.g. '*.nd2','*.czi','*.tif')")
@click.option('--model',        required=True,  type=Path, help="Cellpose model folder or .pth")
@click.option('--out-root',     required=True,  type=Path, help="Root output folder")
@click.option('--gpu/--cpu',    default=True,      help="Toggle GPU acceleration")
@click.option('--do-qc/--no-qc', default=False,    help="Save segmentation QC images")
@click.option('--invert/--no-invert', default=False, help="Invert image intensities before segmentation")
@click.option('--blur-sigma',   default=None,   type=float, help="Optional gaussian blur sigma before segmentation")

def segment(raw_dir, denoised_dir, pattern, model, out_root, gpu, do_qc, invert, blur_sigma):
    """
    1. Build file index (raw + optional denoised)  
    2. Run Cellpose segmentation + EDT  
    3. (optionally) save QC overlays  
    """
    # 1. index
    idx_csv = out_root / "fileList.csv"
    fi = FileIndex(raw_dir, denoised_dir, idx_csv)
    df = fi.build(pattern)

    from .segment import Segmenter
    # 2. segment + edt
    seg = Segmenter(model_path=model, gpu=gpu, invert=invert, blur_sigma=blur_sigma)
    seg.run_on_directory(
        file_index=df,
        out_seg_dir=out_root/"segmentation",
        out_edt_dir=out_root/"edt",
        qc=do_qc,
        qc_func=lambda vol, m, pid, t: plot_segmentation_slices(
            img_5d=vol,
            masks=m,
            out_dir=out_root/"qc",
            sample_id=pid,
            timepoint=t
        )
    )

@cli.command()
@click.option('--raw-dir',  required=True, type=Path, help="Same raw-dir as above")
@click.option('--seg-dir',  required=True, type=Path, help="Output segmentation folder")
@click.option('--out-root', required=True, type=Path, help="Same out-root as above")
@click.option('--spot-ch',  default=1, show_default=True, help="Spot channel index")
@click.option('--nuc-ch',   default=0, show_default=True, help="Nucleus channel index")

def analyze(raw_dir, seg_dir, out_root, spot_ch, nuc_ch):
    """
    1. Compute distance–intensity profiles per position  
    2. Write per-position CSVs and pickles  
    3. Write combined summary tables  
    """
    da = DistanceAnalyzer(
        raw_dir=raw_dir,
        seg_dir=seg_dir,
        output_dist_dir=out_root/"dist",
        output_nuclei_dir=out_root/"nuclei",
        spot_channel=spot_ch,
        nuc_channel=nuc_ch
    )

    # run per‐position
    logger.info("Starting full distance analysis")
    for p in raw_dir.glob("*"):
        if p.suffix.lower() in (".nd2", ".tif") and not p.stem.endswith("_max"):
            logger.info(f"Submitting analysis for {p.stem}")
            da.run_for_position(p.stem)

    # aggregate
    logger.info("Collecting per-position CSVs into one summary")
    da.collect_all_nuclei_csv(
        df_index=[p.stem for p in raw_dir.glob("*") if p.suffix.lower() in (".nd2", ".tif") and not p.stem.endswith("_max")]
    )
    logger.info("Collecting all distance profiles into one pickle")
    da.collect_all_dist_profiles(
        df_index=[p.stem for p in raw_dir.glob("*") if p.suffix.lower() in (".nd2", ".tif") and not p.stem.endswith("_max")]
    )
    logger.info("Distance analysis complete")



    
if __name__ == "__main__":
    cli()
