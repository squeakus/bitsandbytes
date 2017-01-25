var map

function removeTopLayer(){
    var layers = map.getLayers();
    layers.pop();
}

function swapTopLayer(){
    var layers = map.getLayers();
    var topLayer = layers.removeAt(2);
    layers.insertAt(1, topLayer);
}


function init(){
    console.log("in init!");
    map = new ol.Map({
        target:'map',
        renderer:'canvas',
        view: new ol.View({
            //projection: 'EPSG:29902',
            projection: 'EPSG:900913',
            //dublincenter:[-696887.677314,7047964.64674],
	    center:[-8015003.33712,4160979.44405],
            zoom:5
        })
    });

    var styleCache = {};
    var geoLayer = new ol.layer.Vector({
	source : new ol.source.Vector({
	    projection : 'EPSG:900913',
	    url : './myGeoJson.json',
            format: new ol.format.GeoJSON()
	}),
	style : function(feature, resolution) {
	    var text = resolution < 5000 ? feature.get('name') : '';
	    if (!styleCache[text]) {
			styleCache[text] = [new ol.style.Style({
			    fill : new ol.style.Fill({
				color : 'rgba(255, 255, 255, 0.1)'
			    }),
			    stroke : new ol.style.Stroke({
				color : '#319FD3',
				width : 1
			    }),
			    text : new ol.style.Text({
				font : '12px Calibri,sans-serif',
				text : text,
				fill : new ol.style.Fill({
				    color : '#000'
				}),
				stroke : new ol.style.Stroke({
				    color : '#fff',
				    width : 3
				})
			    }),
			    zIndex : 999
			})];
	    }
	    return styleCache[text];
	}
    });
    map.addLayer(geoLayer);
    
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

    var vectorLayer_2 = new ol.layer.Tile({
	source: new ol.source.TileWMS({
	    preload: Infinity,
	    url: 'http://felek.cns.umass.edu:8080/geoserver/wms',
	    serverType:'geoserver',
	    params:{
		'LAYERS':"Streams:Deposition_of_Nitrogen", 'TILED':true
	    }
	})
    });	
    map.addLayer(vectorLayer_2);

    //ZoomToExtent
    var myExtentButton = new ol.control.ZoomToExtent({
	extent:undefined
    });
    map.addControl(myExtentButton);

    map.on('singleclick', function(evt){
	var coord = evt.coordinate;
	var transformed_coordinate = ol.proj.transform(coord, "EPSG:900913", "EPSG:4326");
	console.log(transformed_coordinate);
    })
}
