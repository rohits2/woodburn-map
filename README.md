# Woodburn Map Maker

This is code to generate aerial maps of urban buildings to be fed into a laser engraver and burned onto pieces of wood.  The raw data is from OpenStreetMap via Geofabrik shapefiles, which are then transformed using the Mercator projection (EPSG 3395) and rasterized.  A few lines of text (title, coordinates of the centerpoint, optional subtitle) are added to the bottom.  The final image uses the red channel of a PNG for the engraved rasters, and the green channel to render a border around the image to laser cut the whole board out.

## Example Raster
![SF Raw Raster](maps/san-francisco-new.png)

## Example Engraved Result

Side notes: 
 - This plywood is bad
 - This stain is worse
 - The panel as a whole is ~22"x34"

![Real Engraving](images/result.jpg)