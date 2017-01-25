import web

render = web.template.render('templates/')

urls = (
    #'/(.*)', 'index'
    '/', 'index'
)

class index:
    def GET(self):
        i = web.input(name=None)
        print "jsonp key:", i.jsonp_callback
        web.header('Content-Type', 'text/javascript')
        
        return "{name : 'joe'}"
        #return render.index(i.name)


if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()
