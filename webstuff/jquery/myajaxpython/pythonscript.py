import web
render = web.template.render('templates/')

urls = (
    #'/(.*)', 'index'
    '/', 'index'
)

class index:
    def GET(self):
        i = web.input(name=None)
        print "I GOT CALLED"
        return render.index(i.name)

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()
