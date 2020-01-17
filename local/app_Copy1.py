from flask import (
    Flask,
    render_template,
    jsonify,
    request)
from IPython.display import HTML
import pandas as pd
pd.set_option('display.max_colwidth', -1)

import plotly as py
import plotly.graph_objs as go
import plotly.express as px
py.offline.init_notebook_mode(connected = True)

import requests
# import json
# import math
# from bs4 import BeautifulSoup as bs
# import re

from sklearn import linear_model
import statsmodels.api as sm


app = Flask(__name__)

#FUNCTIONS=======================================================

#========================================================================


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        typ = request.form["typ"]
        typ1 = typ
        srch = request.form["srch"]
        radius = request.form["radius"]
        if radius == "":
            radius = 15000
        pois = request.form["pois"]
        
        if typ == "Address":
            addresses = srch.split(",")
            srch = findzip(srch)
            typ = "Zip"
        zillow_rent = pd.read_csv("https://raw.githubusercontent.com/yundk7/area_lookup_heroku/master/local/Zip_ZriPerSqft_AllHomes.csv")#for rent
        zillow_rent = zillowELT(zillow_rent,typ,srch)
        
        rent_html = zillowplot(zillow_rent)

        zips = zillow_rent.columns.str.replace(" ","").str.split(":")
        zips = [item[1] for item in zips]
        seperator = ","
        zips = seperator.join(zips)
        crime = crimerate(zips)
        crime_html = crime.to_html()
        
        zillow_sales = pd.read_csv("https://raw.githubusercontent.com/yundk7/area_lookup_heroku/master/local/Zip_MedianListingPricePerSqft_AllHomes.csv")
        zillow_sales = zillowELT(zillow_sales,typ,srch)
        
        #returning last record of sales data for zips looking at
        sales_table = zillow_sales.copy()
        header = sales_table.columns.str.replace(" ","").str.split(":")
        sales_table.columns = [item[1] for item in header]
        sales_table = pd.DataFrame(sales_table.iloc[-1].T).reset_index()
        sales_table.columns = ["zip","sales"]
        
        
        sales_html = zillowplot(zillow_sales)
        
        #to find ratio, find intersection of zip from rent and sales
        rent_zip = zillow_rent.columns
        sales_zip = zillow_sales.columns
        ratio_zip = [value for value in rent_zip if value in sales_zip]
        ratio_df = pd.DataFrame()
        for z in ratio_zip:
            ratio_df[z] = (zillow_rent[z]*12)/(zillow_sales[z])*100
        ratio_html = zillowplot(ratio_df)
        
        zips = ratio_df.columns.str.replace(" ","").str.split(":")
        zips = [item[1] for item in zips]
        if pois != "":
            if typ1 != "Address":
                API_record = google_usa(zips,pois)
#                 API_record["link"]=API_record["link"].apply(lambda x: '<a href="https://www.google.com/maps/place/?q=place_id:{0}">link</a>'.format(x))
            if typ1 == "Address":
                API_record = google_geo(addresses,pois,radius)
#                 API_record["link"]=API_record["link"].apply(lambda x: '<a href="https://www.google.com/maps/place/?q=place_id:{0}">link</a>'.format(x))
#             API_pivot = API_record.pivot_table(index = "zip",columns = ["poi"], values=["reviews"],aggfunc=["mean"])["reviews"].reset_index()
            API_df = API_record[API_record["poi"]!="YOU ARE HERE"]
            count_df = API_df.pivot_table(fill_value=0,index = "zip",columns = ["poi"], values="reviews",aggfunc=["mean","count"])["count"].reset_index()
            mean_df = API_df.pivot_table(fill_value=0,index = "zip",columns = ["poi"], values="reviews",aggfunc=["mean","sum"])["mean"].reset_index()
            API_pivot = pd.merge(count_df,mean_df,on="zip",suffixes=["_count","_mean"])
            regression = pd.merge(sales_table,API_pivot,on = "zip")
            regression = pd.merge(regression,crime, on = "zip")
            regression.to_csv("regression.csv",index=False)
            
            x = regression.columns[2:]
            y = regression.columns[1]
            X = regression[x]
            Y = regression[y]

            lm = linear_model.LinearRegression()
            lm.fit(X, Y)
            r_sq = lm.score(X,Y)
            
            intercept = lm.intercept_
            coef = list(lm.coef_)
            regr_df = pd.DataFrame()
            index = ["R^2","intercept"] + list(x)
            coef = [r_sq,intercept] + coef
            regr_df["index"] = index
            regr_df["coef"] = coef
            
#             x = regression.columns[2:]
#             y = regression.columns[1]
#             lm=sm.OLS(Y,X)
#             results = lm.fit()
#             regr_df = results.summary()
            
    
            
            geo_plot = plotly_geo(API_record)
        else:
            API_record = pd.DataFrame()
            geo_plot = ""
        return(crime_html
               +"RENT: $/sqft"
               +rent_html
               +"SALES: $/sqft"
               +sales_html
               +"ROI (PER YERAR: ROI = RENT*12/SALES*100)"
               +ratio_html
               +API_record.to_html(escape=False)
               +geo_plot
               +regression.to_html()
               +"\n Regression Summary: \n R^2 shows credibility of result \n Larger the absolute value of coefficient, bigger the influence on outcome"
               +regr_df.to_html()
              )
    
    return render_template("form.html")

@app.route("/kakao", methods=["GET", "POST"])
def kakao():
    if request.method == "POST":
        center = request.form["center"]
        pois = request.form["pois"]
        radius = request.form["radius"]
        df = kakao_api(center,pois,radius)
        plot = plotly_geo(df)
        
        return(df.to_html(escape=False)+ plot)
    return render_template("form_kakao.html")

@app.route("/ggl", methods=["GET", "POST"])
def ggl():
    if request.method == "POST":
        center = request.form["center"]
        center = center.split(",")
        pois = request.form["pois"]
        radius = request.form["radius"]
        df = google_geo(center,pois,radius)
        plot = plotly_geo(df)
        return(df.to_html(escape=False)+ plot)
    return render_template("form_ggl.html")

if __name__ == "__main__":
    app.run(debug=True)