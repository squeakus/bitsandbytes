import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as pyplt

DATE_INDEX = -2
review_data = pd.read_csv('./pitchfork_review_data.csv', parse_dates=[DATE_INDEX])
print "\ndata structure:"
print review_data
print "\nscore"
print review_data['score'].describe()
print "\nBest New Music"
print review_data[review_data.accolade == ' Best New Music '].describe()
# head() will give us the five lowest scoring reviews
print "\nhighest reviews"
print review_data[review_data.accolade == ' Best New Music '].sort('score').head()    
# tail() will give us the five highest
print "\nworst reviews"
review_data[review_data.accolade == ' Best New Music '].sort('score').tail()


# pyplt.hist(review_data['score'])
# pyplt.show()
# pyplt.hist(review_data['score'], bins=20)
# pyplt.show()
# pyplt.hist(review_data['score'], bins=50)
# pyplt.show()
# daily_data = review_data.groupby("publish_date")['score'].mean()
# daily_data.plot()
# pyplt.show()
# monthly_data = daily_data.resample('M', how='mean')
# monthly_data.plot()
# pyplt.show()
# monthly_data.plot()
# monthly_frame = monthly_data.reset_index()
# total_points = len(monthly_data)
# model = pd.ols(y=monthly_frame[0], x=pd.DataFrame(range(0, total_points)), intercept=True)

reviewer_data = review_data.groupby('reviewer')['score']
aggregated_reviewers = reviewer_data.mean()
aggregated_reviewers.sort('mean')
