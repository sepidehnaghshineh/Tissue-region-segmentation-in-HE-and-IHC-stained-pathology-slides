import openslide
import PIL


slide = openslide.OpenSlide('/mnt/Data1TB/data_snaghshineh/TCGA/testing/5c1216f3-19ec-4d3c-9bb0-9bd740b79f62/TCGA-AR-A1AW-01Z-00-DX1.E527CA46-D83F-4055-8C7E-AEFEF13C1E29.svs')
dimensions = slide.level_dimensions[-1]
print(dimensions)
print(slide.level_count)

image = slide.read_region((0,0),3,dimensions)
image = image.convert("RGB")


image.show()
image.save('/home/snaghshineh/Documents/TissueSegmentation/DATA_PREP/tumbnail-results/TCGA-AR-A1AW-01Z-00-DX1.png')
