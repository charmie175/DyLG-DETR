from tidecv import TIDE, datasets

tide = TIDE()
tide.evaluate(datasets.COCO('/root/autodl-tmp/visdrone/val_fixed.json'), datasets.COCOResult('/root/autodl-tmp/output/bbox.json'), mode=TIDE.BOX) # Use TIDE.MASK for masks
tide.summarize()  # Summarize the results as tables in the console
tide.plot()       # Show a summary figure. Specify a folder and it'll output a png to that folder.