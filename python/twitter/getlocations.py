import urllib2, json, pprint
user = "johnmarksuave"

user_friends = urllib2.urlopen('https://api.twitter.com/1/friends/ids.json?cursor=-1&screen_name='+user+'&stringify_ids=true')
resultdict = json.load(user_friends)

for result in resultdict:
    print result

print "ids:",resultdict['ids']
id_list = resultdict['ids']

for user_id in id_list:
    query_string = 'https://api.twitter.com/1/users/show.json?id=' + user_id
    user_info = urllib2.urlopen(query_string)
    userdict = json.load(user_info) 
    print userdict['name']
    if userdict['location'] == "":
        print "empty"
    else:
        print  userdict['location']
    #for result in userdict:
    #    print result
   #    timeline = "https://api.twitter.com/1/statuses/user_timeline.json?screen_name="+userdict['name']+"&count=2"
