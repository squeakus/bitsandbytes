from elaphe import barcode
barcode('ISBN', '977147396801', options=dict(includetext=True)).show()
barcode('ISBN', '000034334891').show()
barcode('Ean13', '977147396801').show()
barcode('raw', '000011116801').show()

barcode('qrcode',
        "hello itsa me mario",
        options=dict(version=9, eclevel='M'), 
        margin=10, data_mode='8bits').show()
