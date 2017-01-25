var map

function init(){
    console.log("in my init!");
    
    // taken from epsg.io
    proj4.defs("EPSG:29902","+proj=tmerc +lat_0=53.5 +lon_0=-8 +k=1.000035 +x_0=200000 +y_0=250000 +ellps=mod_airy +towgs84=482.5,-130.6,564.6,-1.042,-0.214,-0.631,8.15 +units=m +no_defs");

    var irish65proj = new ol.proj.Projection({
	code: 'EPSG:29902',
	extent: [21861.71, 18286.84, 368431.41, 468039.71]
	//extent: [485869.5728, 76443.1884, 837076.5648, 299941.7864]
    });

    //var lonlat = ol.proj.transform([-6.260243,53.349782],"WGS84", "EPSG:900913");
var lonlat = ol.proj.transform([315904, 234671],irish65proj, "EPSG:900913");

    map = new ol.Map({
        target:'map',
        renderer:'canvas',
        view: new ol.View({
            //projection: irish65proj,
	    projection: 'EPSG:900913',
	    //projection:'EPSG:4326', 
	    //dublin centre
            center: lonlat,
	    //irish grid centre
	    //center:[315904, 234671],
            zoom:15
        })
    });

    
    var newLayer = new ol.layer.Tile({
	source: new ol.source.OSM()
    });
    map.addLayer(newLayer);

    var vectorLayer = new ol.layer.Tile({
	source: new ol.source.TileWMS({
	    preload: Infinity,
	    url: 'http://felek.cns.umass.edu:8080/geoserver/wms',
	    serverType:'geoserver',
	    params:{
		'LAYERS':"Streams:Developed", 'TILED':true
	    }
	})
    });
    vectorLayer.setOpacity(.3);
    map.addLayer(vectorLayer);
}
