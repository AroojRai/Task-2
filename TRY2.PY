import pandas as pd
from sklearn.linear_model import LinearRegression
data= {
    "students":["a","b","c","d","e"],
    "hours":[5,2,7,3,6],
    "percentage(%)":[75,60,85,65,80]
}

df = pd.DataFrame(data)
print(df)
x= df[["hours"]]          #this is idependent variable thats why used double brackets 
y= df["percentage(%)"]
# model training step
model= LinearRegression() # empty brackets means that we are calling that function 
model.fit(x,y)

#model evaluation 
r_squared= model.score(x,y)
print(r_squared)

accuracy= model.score(x,y)

print(f"the accuracy of model is {accuracy*100:.2f}")
new= pd.DataFrame({"hours":4})
prediction= model.predict(new)
print(prediction)