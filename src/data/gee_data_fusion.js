/**
 * ===============================================================================
 * Project: Multimodal Satellite Data Fusion for Enhanced Land Cover Classification
 * Target Area: Bengaluru, Karnataka, India
 * Platform: Google Earth Engine (JavaScript Code Editor API)
 * Datasets: Sentinel-1 GRD SAR, Sentinel-2 L2A SR, ESA WorldCover v200
 * ===============================================================================
 * 
 * DESCRIPTION:
 * This script prepares, masks, composites, harmonizes, and visualizes multimodal
 * satellite imagery for land cover classification research.
 * 
 * REQUIREMENTS COVERED:
 * 1. Define study area around Bengaluru, Karnataka, India.
 * 2. Collect Sentinel-1 GRD data (VV, VH, IW mode).
 * 3. Create median Sentinel-1 composite.
 * 4. Collect Sentinel-2 Surface Reflectance (COPERNICUS/S2_SR_HARMONIZED).
 * 5. Select bands: B2, B3, B4, B8, B11.
 * 6. Apply QA60 cloud masking to Sentinel-2.
 * 7. Create median Sentinel-2 composite.
 * 8. Load ESA WorldCover v200 dataset.
 * 9. Clip all datasets to ROI.
 * 10. Display all datasets on GEE Interactive Map.
 * 11. Harmonize spatial resolutions to 10m grid.
 * 12. Comprehensive inline comments for every section.
 * ===============================================================================
 */

// ===============================================================================
// SECTION 1: STUDY AREA DEFINITION (BENGALURU, KARNATAKA, INDIA)
// ===============================================================================

// Define rectangular ROI around Bengaluru metropolis and surrounding rural fringe
// Coordinates: [Min Longitude, Min Latitude, Max Longitude, Max Latitude]
var roi = ee.Geometry.Rectangle([77.45, 12.80, 77.75, 13.15]);

// Center GEE interactive map view on Bengaluru ROI at zoom level 11
Map.centerObject(roi, 11);
Map.style().set('cursor', 'crosshair');


// ===============================================================================
// SECTION 2: SENTINEL-1 SAR GRD DATA COLLECTION & FILTERING
// ===============================================================================

// Define temporal window for compositing (1-year baseline for clear seasonal median)
var startDate = '2023-01-01';
var endDate = '2023-12-31';

// Query Sentinel-1 Ground Range Detected (GRD) C-Band SAR dataset
// Filters: ROI, Date Range, IW (Interferometric Wide) Mode, Dual Polarization (VV & VH)
var s1Collection = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(roi)
  .filterDate(startDate, endDate)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));

// Print Sentinel-1 collection metadata to Console for validation
print('=== SENTINEL-1 SAR METADATA ===');
print('Sentinel-1 Image Count:', s1Collection.size());


// ===============================================================================
// SECTION 3: SENTINEL-1 MEDIAN COMPOSITE & BAND RATIO CREATION
// ===============================================================================

// Select VV and VH backscatter coefficients (in dB), calculate median composite, and clip to ROI
var s1Median = s1Collection
  .select(['VV', 'VH'])
  .median()
  .clip(roi);

// Calculate cross-polarization ratio (VV / VH) to highlight structural roughness and moisture
var vvVhRatio = s1Median.select('VV').divide(s1Median.select('VH')).rename('VV_VH');

// Combine VV, VH, and VV/VH ratio into a 3-band SAR composite
var s1Composite = s1Median.addBands(vvVhRatio);


// ===============================================================================
// SECTION 4, 5, 6 & 7: SENTINEL-2 OPTICAL SR COLLECTION, MASKING & COMPOSITING
// ===============================================================================

/**
 * Cloud and Cirrus Masking Function for Sentinel-2 Surface Reflectance (Level-2A).
 * Uses QA60 bitmask band: Bit 10 = Opaque Clouds, Bit 11 = Cirrus Clouds.
 * Normalizes reflectance values by dividing by 10,000.
 */
function maskS2Clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  
  // Keep pixels where both cloud and cirrus bit flags are 0 (clear conditions)
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
    
  return image.updateMask(mask).divide(10000.0);
}

// Query Sentinel-2 Surface Reflectance Harmonized dataset
var s2Collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi)
  .filterDate(startDate, endDate)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(maskS2Clouds);

// Print Sentinel-2 collection metadata to Console for validation
print('=== SENTINEL-2 OPTICAL METADATA ===');
print('Sentinel-2 Image Count:', s2Collection.size());

// Select requested bands: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR 1)
var s2Bands = ['B2', 'B3', 'B4', 'B8', 'B11'];

// Compute median composite of optical reflectance bands and clip to ROI
var s2Median = s2Collection
  .select(s2Bands)
  .median()
  .clip(roi);


// ===============================================================================
// SECTION 8 & 9: ESA WORLDCOVER v200 LAND COVER DATASET
// ===============================================================================

// Load ESA WorldCover 10m v200 (2021 land cover product)
var worldCoverCollection = ee.ImageCollection('ESA/WorldCover/v200');
var worldCover = worldCoverCollection.first().select('Map').clip(roi);

print('=== ESA WORLDCOVER v200 METADATA ===');
print('ESA WorldCover Product:', worldCover);


// ===============================================================================
// SECTION 11: SPATIAL RESOLUTION ALIGNMENT & HARMONIZATION (10M GRID)
// ===============================================================================

// Retrieve native 10m spatial reference projection from Sentinel-2 Blue band (B2)
var target10mProj = s2Median.select('B2').projection();

// Resample Sentinel-2 (using bilinear interpolation for 20m SWIR B11) onto 10m grid
var s2Resampled = s2Median.resample('bilinear').reproject({
  crs: target10mProj,
  scale: 10
});

// Reproject Sentinel-1 SAR composite onto exact 10m CRS grid for multimodal fusion co-registration
var s1Resampled = s1Composite.reproject({
  crs: target10mProj,
  scale: 10
});


// ===============================================================================
// SECTION 10: INTERACTIVE MAP VISUALIZATION
// ===============================================================================

// 1. Study Area ROI Boundary Layer (Red Outline)
var roiOutline = ee.Image().byte().paint({
  featureCollection: ee.FeatureCollection([ee.Feature(roi)]),
  color: 1,
  width: 2
});
Map.addLayer(roiOutline, {palette: ['FF0000']}, '0. Study Area ROI Boundary');

// 2. Sentinel-2 True Color RGB Composite (B4 = Red, B3 = Green, B2 = Blue)
var s2TrueColorVis = {
  bands: ['B4', 'B3', 'B2'],
  min: 0.0,
  max: 0.3,
  gamma: 1.2
};
Map.addLayer(s2Resampled, s2TrueColorVis, '1. Sentinel-2 True Color (RGB: B4/B3/B2)');

// 3. Sentinel-2 False Color Infrared Composite (B8 = NIR, B4 = Red, B3 = Green)
var s2FalseColorVis = {
  bands: ['B8', 'B4', 'B3'],
  min: 0.0,
  max: 0.4,
  gamma: 1.2
};
Map.addLayer(s2Resampled, s2FalseColorVis, '2. Sentinel-2 False Color (NIR/Red/Green)');

// 4. Sentinel-2 SWIR Composite (B11 = SWIR1, B8 = NIR, B4 = Red) - Urban & Soil distinction
var s2SwirVis = {
  bands: ['B11', 'B8', 'B4'],
  min: 0.0,
  max: 0.45
};
Map.addLayer(s2Resampled, s2SwirVis, '3. Sentinel-2 SWIR Composite (B11/B8/B4)');

// 5. Sentinel-1 SAR RGB Composite (Red: VV, Green: VH, Blue: VV/VH Ratio)
var s1SarVis = {
  bands: ['VV', 'VH', 'VV_VH'],
  min: [-20, -25, 1],
  max: [0, -5, 15]
};
Map.addLayer(s1Resampled, s1SarVis, '4. Sentinel-1 SAR Composite (VV/VH/Ratio)');

// 6. ESA WorldCover 2021 10m Land Cover Map
var worldCoverVis = {
  bands: ['Map']
};
Map.addLayer(worldCover, worldCoverVis, '5. ESA WorldCover 2021 (Land Cover Map)');


// ===============================================================================
// SECTION 12: DATASET VALIDATION & MULTIMODAL STACK CONSTRUCT
// ===============================================================================

// Stack all harmonized layers into a single multi-band multimodal validation image
var multimodalStack = s2Resampled.addBands(s1Resampled).addBands(worldCover);

print('=== MULTIMODAL FUSION STACK VALIDATION ===');
print('Fused Multimodal Band Names:', multimodalStack.bandNames());
print('Projection Info (10m Resolution):', multimodalStack.select('B2').projection());
print('Pixel Scale (Meters):', multimodalStack.select('B2').projection().nominalScale());

// Instructions for User in Console
print('=== VALIDATION & INSPECTION INSTRUCTIONS ===');
print('1. Use the "Inspector" tab in the top-right of the GEE Code Editor.');
print('2. Click anywhere in the Bengaluru region on the map.');
print('3. Verify that pixel values for optical (B2, B3, B4, B8, B11), SAR (VV, VH, VV_VH), and ESA WorldCover (Map) are aligned and displayed simultaneously.');
