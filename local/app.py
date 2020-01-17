from flask import (
    Flask,
    render_template,
    jsonify,
    request)
from IPython.display import HTML
import numpy as np
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
from sqlalchemy import create_engine

# from sklearn import linear_model
import statsmodels.api as sm

app = Flask(__name__)

#FUNCTIONS=======================================================
def zillowELT(df,zips_list):
    df.rename(columns = {"RegionName":"zip"},inplace=True)
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    df = df[df["zip"].isin(zips_list)]
    df.set_index("zip",inplace=True)
    #cut upto the column "sizerank"
    cut = list(df.columns).index("SizeRank")+1
    df = df.iloc[:,cut:]
    df.columns = pd.to_datetime(df.columns,format = "%b-%y",errors = "coerce")
    return(df)

def zillowplot(df):
    traces = []
    for i in range(1,df.shape[1]+1):
        x = df.index
        y= df[df.columns[i-1]]
        trace = go.Scatter(x=x,y=y,
        mode = 'lines',
        name = df.columns[i-1])
        traces.append(trace)
    fig = go.Figure(data=traces)
    return(py.offline.plot(fig,output_type="div"))

def dfs_shape_merge(df1,df2):
    #to get ratio, first put rent and sales dataframe into same size with matching index and columns
    #.T to transpose
    df1_T = df1[df1.index.isin(list(df2.index))].T
    df1_T = df1_T[df1_T.index.isin(list(df2.T.index))]
    df2_T = df2[df2.index.isin(list(df1.index))].T
    df2_T = df2_T[df2_T.index.isin(list(df1.T.index))]
    return([df1_T,df2_T])

#will merge dfs respect to index values
def merge_dfs(df_list):
    for n in range(0,len(df_list)):
        if n == 0:
            df = df_list[0]
        else:
            df = pd.merge(df,df_list[n],left_index=True,right_index=True)
    return(df)

def regression(df):
    x = df.columns[1:]
    y = df.columns[0]
    X = df[x]
    Y = df[y]

#     lm = linear_model.LinearRegression()
#     lm.fit(X, Y)
#     r_sq = lm.score(X,Y)

#     intercept = lm.intercept_
#     coef = list(lm.coef_)
#     regr_df = pd.DataFrame()
#     index = ["R^2","intercept"] + list(x)
#     coef = [r_sq,intercept] + coef
#     regr_df["index"] = index
#     regr_df["coef"] = coef
#     return(regr_df)

#     X = sm.add_constant(X) # adding a constant
    X.insert(0,"intercept",1)
    model = sm.OLS(Y, X).fit()
    predictions = model.predict(X) 

    print_model = model.summary2().tables
    return(print_model)

def google_zip_df(df,pois):
    records = pd.DataFrame()
    gkey = "AIzaSyA-Rjp6nOeJp6815Xt1Kkuxc5XKMiKl_yA"
    pois = pois.split(",")
    for n in range(0,len(df)):
        center_zip = df.index[n]
        center_coordinates = df.coordinates[n].replace(" ","")
        lat = center_coordinates.split(",")[0]
        lng = center_coordinates.split(",")[1]
        radius = df.radius[n]*1600
        center_df = pd.DataFrame({"zip":[center_zip],"poi":["YOU ARE HERE"],"name":["YOU ARE HERE"],"address":[f"zip:{center_zip}"],"Y":[float(lat)],"X":[float(lng)]})
        records = records.append(center_df)
        
        #get radius of each zip
        
        
        for poi in pois:
            params = {
                "location": center_coordinates,
                "keyword": poi,
                "radius": radius,
            #     "type": target_type,
                "key": gkey
            }

            # base url
            base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

            # run a request using our params dictionary
            response = requests.get(base_url, params=params)
            places_data = response.json()
            n=0
            # while int(n) > len(places_data):
            while int(n) < len(places_data["results"]):
                try:
                    price=places_data["results"][int(n)]["price_level"]
                except KeyError:
                    price = np.nan
                try:
                    link=places_data["results"][int(n)]["place_id"]
                except KeyError:
                    link = "NaN"
                try:
                    score = places_data["results"][int(n)]["rating"]
                except KeyError:
                    score = np.nan
                try:
                    reviews = int(places_data["results"][int(n)]["user_ratings_total"])
                except KeyError:
                    reviews = np.nan
                try:
                    lat1 = places_data["results"][int(n)]["geometry"]["location"]["lat"]
                except KeyError:
                    lat1 = np.nan
                try:
                    lng1 = places_data["results"][int(n)]["geometry"]["location"]["lng"]
                except KeyError:
                    lng1 = np.nan
                content = pd.DataFrame ({"zip":center_zip,
                                         "poi":poi,
                                    "name":[places_data["results"][int(n)]["name"]],
                                    "score":score,
                                     "reviews":reviews,
                                     "price":price,
                                     "link":link,
                                    "address":[places_data["results"][int(n)]["vicinity"]],
                                         "Y":[lat1],
                                         "X":[lng1]
            #                        "distance":distance,
            #                         "drive":duration,
            #                         "public":transit_dur,
            #                         "walk":walk_dur
                                        })
                records = records.append(content)
                n+=1
    records.reset_index(drop = True,inplace = True)
#     records = records[["center","zip","poi","name","score","reviews","price","price","link","address","X","Y"]]
    records["link"]=records["link"].apply(lambda x: '<a href="https://www.google.com/maps/place/?q=place_id:{0}">link</a>'.format(x))
    return(records)

def plotly_geo(df):
    df["size"] = 10
    df.loc[df['poi'] == "YOU ARE HERE", 'size'] = 20
    df.loc[df['poi'] == "YOU ARE HERE", 'reviews'] = 0
    px.set_mapbox_access_token("pk.eyJ1IjoidGl2bWU3IiwiYSI6ImNrMWEwZDVtNDI4Zm4zYm1vY3o3Z25zejEifQ._yTPkj3nXTzor72zIevLCQ")
    hover = ["reviews"]
#     if hover in (df.columns):
#         hover = ["name","reviews"]
#     else:
#         hover = []
    fig = px.scatter_mapbox(df, lat="Y", lon="X", color = "poi", size = "size", hover_name="name",zoom = 13, hover_data = hover)
    fig.update_layout(autosize=True,width=1500,height=750)
        #                           ,margin=go.layout.Margin(l=50,r=50,b=100,t=100,pad=4))
    return(py.offline.plot(fig,output_type="div"))
#========================================================================


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        typ = request.form["typ"]
        srch = request.form["srch"]
        poi = request.form["pois"]
        typ = typ.lower()
        #to take account typing differences such as space and capital letters
        srch = srch.upper().replace(" ","").split(",")
        poi = poi.upper()
        
        #call in information to find what zipcodes are in search area(eg:chicago)
        con = create_engine("sqlite:///zip_data0.sqlite")
        city = pd.read_sql("city",con)
        city = city[city[typ].astype(str).str.replace(" ","").str.upper().isin(srch)]
        #from zip codes obtained, select information needed from each database
        zips = list(city["zip"])
        
        rent = pd.read_csv("Zip_ZriPerSqft_AllHomes.csv")
        rent = zillowELT(rent,zips)
        rent_plt = zillowplot(rent.T)
        
        sales = pd.read_csv("Zip_MedianListingPricePerSqft_AllHomes.csv")
        sales = zillowELT(sales,zips)
        sales_plt = zillowplot(rent.T)
        
        dfs = dfs_shape_merge(rent,sales)
        rent1 = dfs[0]
        sales1 = dfs[1]
        ratio = (rent1 * 12) / sales1 * 100
        ratio_plt = zillowplot(ratio)
        
        #refer to zips in both rent and sales, hence ratio
        #then append necessary data from db
        df = pd.DataFrame(index = ratio.T.index)
        
        #call in crime, density etc from db
        crime = pd.read_sql("crime",con)
        crime = crime[crime["zip"].isin(list(df.index))]
        crime.set_index("zip",inplace=True)
        area = pd.read_sql("area",con)
        area = area[area["zip"].isin(list(df.index))]
        area.set_index("zip",inplace=True)
        
        df = merge_dfs([df,crime,area])
        
        api = google_zip_df(df,poi)
        geo_plt = plotly_geo(api)
        
        API_df = api[api["poi"]!="YOU ARE HERE"]
        API_df[["reviews"]]=API_df[["reviews"]].apply(pd.to_numeric, errors='coerce')
        count_df = API_df.pivot_table(fill_value=0,index = "zip",columns = ["poi"], values="reviews",aggfunc=["mean","count"])["count"].reset_index()
        mean_df = API_df.pivot_table(fill_value=0,index = "zip",columns = ["poi"], values="reviews",aggfunc=["mean","sum"])["mean"].reset_index()
        API_pivot = pd.merge(count_df,mean_df,on="zip",suffixes=["_count","_mean"])
        API_pivot.set_index("zip",inplace=True)
        regr = merge_dfs([df,API_pivot])
        #dropping unnecessary columns
        regr.drop(columns = ["coordinates","area","radius"],inplace=True)
        
        #preparing sqlite for temporarily stored dataframes
        con = create_engine("sqlite:///summary.sqlite")
        api.to_sql("api",con,if_exists="replace",index=False)
        regr.to_sql("pivot",con,if_exists="replace")
        
        #including rent,sales,ratio to regression formula
        #averaging last 5 results of data
        rent_df = rent.iloc[:,-5:]
        rent_df = pd.DataFrame(rent_df.mean(axis=1))
        regr_rent = merge_dfs([rent_df,regr])
        regr_rent = regression(regr_rent)
        regr_rent[0].to_sql("rent0",con,if_exists="replace",index=False)
        regr_rent[1].to_sql("rent1",con,if_exists="replace",index=False)
        
        sales_df = sales.iloc[:,-5:]
        sales_df = pd.DataFrame(sales_df.mean(axis=1))
        regr_sales = merge_dfs([sales_df,regr])
        regr_sales = regression(regr_sales)
        regr_sales[0].to_sql("sales0",con,if_exists="replace",index=False)
        regr_sales[1].to_sql("sales1",con,if_exists="replace",index=False)
        
        ratio_df = ratio.T.iloc[:,-5:]
        ratio_df = pd.DataFrame(ratio_df.mean(axis=1))
        regr_ratio = merge_dfs([ratio_df,regr])
        regr_ratio = regression(regr_ratio)
        regr_ratio[0].to_sql("ratio0",con,if_exists="replace",index=False)
        regr_ratio[1].to_sql("ratio1",con,if_exists="replace",index=False)
        
        #clickable link to summary
        df = pd.DataFrame({"SUMMARY":["/summary"]})
        df["SUMMARY"] = df["SUMMARY"].apply(lambda x: '<a href="{0}">Click to view table only summary(For saving)</a>'.format(x))
        
        return (
            df.to_html(escape=False)+
            "GEO PLOTTING PLACES OF INTEREST"+
            geo_plt+
#             "RESULT OF GOOGLE GEO API SCRAPING"+
#             api.to_html(escape=False)+
            "RENT: $/SQFT"+
            rent_plt+
            "REGRESSION ANALYSIS ON IMPACT OF FACTORS REGARDING RENT"+
            regr_rent[0].to_html()+
            regr_rent[1].to_html()+
            "SALES: $/SQFT"+
            sales_plt+
            "REGRESSION ANALYSIS ON IMPACT OF FACTORS REGARDING SALES"+
            regr_sales[0].to_html()+
            regr_sales[1].to_html()+
            "ROI (PER YERAR: ROI = RENT*12/SALES*100)"+
            ratio_plt+
            "REGRESSION ANALYSIS ON IMPACT OF FACTORS REGARDING ROI"+
            regr_sales[0].to_html()+
            regr_sales[1].to_html()
        )
        
        
    return render_template("form.html")

@app.route("/summary")
def summary():
    con = create_engine("sqlite:///summary.sqlite")
    rent0 = pd.read_sql("rent0",con)
    rent1 = pd.read_sql("rent1",con)
    sales0 = pd.read_sql("sales0",con)
    sales1 = pd.read_sql("sales1",con)
    ratio0 = pd.read_sql("ratio0",con)
    ratio1 = pd.read_sql("ratio1",con)
    api = pd.read_sql("api",con)
    pivot = pd.read_sql("pivot",con)
    
    n = pd.DataFrame().to_html()
    return(
            "Regression analysis on rent"+render_template("n.html")+
        rent0.to_html()+rent1.to_html()+render_template("n.html")+
        "Regression analysis on sales"+render_template("n.html")+
        sales0.to_html()+sales1.to_html()+render_template("n.html")+
        "Regression analysis on ROI"+render_template("n.html")+
        ratio0.to_html()+ratio1.to_html()+render_template("n.html")+
        "Google API results"+render_template("n.html")+
        api.to_html(escape=False)+render_template("n.html")+
        "Pivot table of crime rates, population density, count and mean of POIS"+render_template("n.html")+
        pivot.to_html()
        
    )


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