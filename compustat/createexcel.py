import pyExcelerator as excel

workBookDocument = excel.Workbook()
docSheet1 = workBookDocument.add_sheet("sheet1")
myFont = excel.Font()
myFont.bold = True
myFont.underline = True
myFont.italic = True
#myFont.struck_out = True
myFont.colour_index = 3
myFont.outline = True

myFontStyle = excel.XFStyle()
myFontStyle.font = myFont
docSheet1.row(0).set_style(myFontStyle)
docSheet1.write(0, 0, 'value')
docSheet1.write(0, 1, 'name', myFontStyle)
workBookDocument.save('pytest.xls')
