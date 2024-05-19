import cv2
import numpy as np
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString


def Create_Overlay(image, inverted_Mask, fill=(100, 100, 220, 70)):
    """
    Image: PIL Image
    inverted_Mask: PIL Image (black background, white foreground)
    """
    # Create a new image with the same size as the original image
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    
    # Draw the mask onto the overlay image
    draw = ImageDraw.Draw(overlay)
    draw.bitmap((0, 0), inverted_Mask, fill=fill)
    
    # Blend the overlay onto the original image
    result = Image.alpha_composite(image.convert('RGBA'), overlay)
    
    return result


def mask_to_xml(mask, xml_path, downscale_factor, annotation_type="Polygon", part_of_group="tissue", color="#F4FA58", group_name="tissue", group_color="#64FE2E"):
    # Read the binary mask image

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #there is a bug CHANGE IT

    # Create XML structure
    root = ET.Element("ASAP_Annotations")
    annotations = ET.SubElement(root, "Annotations")

    # Add coordinates to XML for each contour
    for i, contour in enumerate(contours):
        annotation_name = f"Annotation {i}"
        annotation = ET.SubElement(annotations, "Annotation", Name=annotation_name, Type=annotation_type, PartOfGroup=part_of_group, Color=color)

        coordinates = ET.SubElement(annotation, "Coordinates")
        for j, point in enumerate(contour):
            x, y = point[0]
            coordinate = ET.SubElement(coordinates, "Coordinate", Order=str(j), X=str(x*downscale_factor).replace(".",","), Y=str(y*downscale_factor).replace(".",","))

    # Add AnnotationGroups and Group to XML
    annotation_groups = ET.SubElement(root, "AnnotationGroups")
    group = ET.SubElement(annotation_groups, "Group", Name=group_name, PartOfGroup="None", Color=group_color)
    attributes = ET.SubElement(group, "Attributes")

    # Create and save the XML file with indentation
    xml_str = ET.tostring(root, encoding="unicode")
    xml_str = parseString(xml_str).toprettyxml(indent="\t")
    with open(xml_path, "w") as xml_file:
        xml_file.write(xml_str)


def get_eroded_mask(mask, kernel_size=2, iterations=1, inverted=True):
    if inverted:
        mask = cv2.bitwise_not(mask)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=iterations)
    if inverted:
        eroded_mask = cv2.bitwise_not(eroded_mask)
    return eroded_mask