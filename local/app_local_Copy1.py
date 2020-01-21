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
def key(key):
    #API keys
    if key == "gkey":
        key = "AIzaSyCx9a5SZ-y42Wu2fQeqmHsfKFRk4djJsAs"
    elif key == "kkey":
        key = "KakaoAK 8809fcb48aa9900788adbd9f162c6b25"
    elif key == "ptoken":
        key = "pk.eyJ1IjoidGl2bWU3IiwiYSI6ImNrMWEwZDVtNDI4Zm4zYm1vY3o3Z25zejEifQ._yTPkj3nXTzor72zIevLCQ"
    return(key)

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
    df1_T = df1_T[df1_T.index.isin(list(df2.T.index))].T
    df2_T = df2[df2.index.isin(list(df1.index))].T
    df2_T = df2_T[df2_T.index.isin(list(df1.T.index))].T
    return([df1_T,df2_T])

#will merge dfs respect to index values
def merge_dfs(df_list):
    for n in range(0,len(df_list)):
        if n == 0:
            df = df_list[0]
        else:
            df = pd.merge(df,df_list[n],left_index=True,right_index=True)
    return(df)

#will merge outer
def merge_dfs_out(df_list):
    for n in range(0,len(df_list)):
        if n == 0:
            df = df_list[0]
        else:
            df = pd.merge(df,df_list[n],left_index=True,right_index=True,how="outer")
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
    gkey = key("gkey")
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

def plotly_geo(df,name):
    df["size"] = 10
    df.loc[df['poi'] == "YOU ARE HERE", 'size'] = 20
    df.loc[df['poi'] == "YOU ARE HERE", 'reviews'] = 0
    plotly_token = key("ptoken")
    px.set_mapbox_access_token(plotly_token)
    hover = ["reviews"]
#     if hover in (df.columns):
#         hover = ["name","reviews"]
#     else:
#         hover = []
    fig = px.scatter_mapbox(df, lat="Y", lon="X", color = "poi", size = "size", hover_name="name",zoom = 13, hover_data = hover)
    fig.update_layout(autosize=True,width=1500,height=750)
        #                           ,margin=go.layout.Margin(l=50,r=50,b=100,t=100,pad=4))
    return(py.offline.plot(fig,filename=f"templates/demo/{name}.html"))#,output_type="div"))
#========================================================================
@app.route("/")
def home():
    #loading homepage = preparing database
    #for USA database
    con = create_engine("sqlite:///us_db.sqlite")
    rent = pd.read_csv("https://raw.githubusercontent.com/yundk7/area_lookup_heroku/master/local/Zip_ZriPerSqft_AllHomes.csv")
    rent.to_sql("zillow_rent",con,if_exists = "replace", index=False)
    
    sales = pd.read_csv("https://raw.githubusercontent.com/yundk7/area_lookup_heroku/master/local/Zip_MedianListingPricePerSqft_AllHomes.csv")
    sales.to_sql("zillow_sales",con,if_exists = "replace", index=False)
    
    city = pd.read_csv("https://raw.githubusercontent.com/yundk7/area_lookup_heroku/master/local/zip_code_city.csv",dtype={"zip":"str"})
    city.to_sql("city",con,if_exists = "replace", index=False)
    
    crime = pd.read_csv("https://raw.githubusercontent.com/yundk7/area_lookup_heroku/master/local/zip_crime.csv",dtype={"zip":"str"})
    crime.to_sql("crime",con,if_exists = "replace", index=False)
    
    area = pd.read_csv("https://raw.githubusercontent.com/yundk7/area_lookup_heroku/master/local/zip_area.csv",dtype={"zip":"str"})
    area.to_sql("area",con,if_exists = "replace", index=False)
    
    df=pd.DataFrame()
    df["Page"] = [
        "/",
        "/us_local"
    ]
    
    df["Content"] = [
        "Here at the home page, data is stored in temporary sqlite database",
        "With data scraped and gathered through Census bureau, Google API, Zillow Datasets, analyze local amenities and its impact on real estate value"
    ]
    df["Page"] = df["Page"].apply(lambda x: '<a href="{0}">{0}</a>'.format(x))
    
    return(df.to_html(escape=False))

@app.route("/us_local", methods=["GET", "POST"])
def us_local():
    if request.method == "POST":
        typ = request.form["typ"]
        srch = request.form["srch"]
        poi = request.form["pois"]
        name = request.form["name"]
        if name == "":
            name = "nan"
        name = name.lower()
        name = name.replace(" ","_")
        
        con_search = create_engine("sqlite:///search.sqlite")
        df = pd.DataFrame({"Title":["Type","Search","Place of Interests"],"Input":[typ,srch,poi]})
        df.to_sql("search", con_search, if_exists="replace", index=False)

        typ = typ.lower()
        #to take account typing differences such as space and capital letters
        srch = srch.upper().replace(" ","").split(",")
        poi = poi.upper()

        #call in information to find what zipcodes are in search area(eg:chicago)
        con_us = create_engine("sqlite:///us_db.sqlite")
        city = pd.read_sql("city",con_us)
        city = city[city[typ].astype(str).str.replace(" ","").str.upper().isin(srch)]
        #from zip codes obtained, select information needed from each database
        zips = list(city["zip"])

        #         rent = pd.read_csv("https://raw.githubusercontent.com/yundk7/area_lookup_heroku/master/local/Zip_ZriPerSqft_AllHomes.csv")
        rent = pd.read_sql("zillow_rent",con_us)
        rent = zillowELT(rent,zips)
        rent_plt = zillowplot(rent.T)

        #         sales = pd.read_csv("https://raw.githubusercontent.com/yundk7/area_lookup_heroku/master/local/Zip_MedianListingPricePerSqft_AllHomes.csv")
        sales = pd.read_sql("zillow_sales",con_us)
        sales = zillowELT(sales,zips)
        sales_plt = zillowplot(sales.T)

        dfs = dfs_shape_merge(rent,sales)
        rent1 = dfs[0]
        sales1 = dfs[1]
        ratio = (rent1 * 12) / sales1 * 100
        ratio_plt = zillowplot(ratio.T)

        # getting recent 5 for regression later
        n = -5
        rent = rent.iloc[:,n:]
        rent = pd.DataFrame({"rent":rent.mean(axis=1)})

        sales = sales.iloc[:,n:]
        sales = pd.DataFrame({"sales":sales.mean(axis=1)})

        ratio = ratio.iloc[:,n:]
        ratio = pd.DataFrame({"roi":ratio.mean(axis=1)})

        # now the zips are outer merge of zips in rent and sales
        zip_rent = pd.DataFrame(rent.index.values)
        zip_sales = pd.DataFrame(sales.index.values)
        zips = list(pd.merge(zip_rent,zip_sales,on=0,how="outer")[0])

        #preparing dataframe for regression with zipcode data, also to be used in google api search
        area = pd.read_sql("area",con_us)
        area = area[area["zip"].isin(zips)]
        crime = pd.read_sql("crime",con_us)

        regr = pd.merge(area,crime,on="zip")
        regr.set_index("zip",inplace=True)

        #since heroku is limited with request time, sampling out 8 zip codes to analyze
        # sample = 8
        # if len(df) < sample:
        #     sample = len(df)
        # df = df.sample(sample)

        #using regression dataframe for reference
        api = google_zip_df(regr,poi)
        # geo_plt = plotly_geo(api)
        plotly_geo(api,name)
        con_sum = create_engine("sqlite:///summary.sqlite")
        api.to_sql("api",con_sum,if_exists="replace",index=False)

        #preparing api results for regression
        API_df = api[api["poi"]!="YOU ARE HERE"]
        API_df[["reviews"]]=API_df[["reviews"]].apply(pd.to_numeric, errors='coerce')
        count_df = API_df.pivot_table(fill_value=0,index = "zip",columns = ["poi"], values="reviews",aggfunc=["count"])["count"].reset_index()
        mean_df = API_df.pivot_table(fill_value=0,index = "zip",columns = ["poi"], values="reviews",aggfunc=["mean"])["mean"].reset_index()
        API_pivot = pd.merge(count_df,mean_df,on="zip",suffixes=["_count","_mean"])
        API_pivot.set_index("zip",inplace=True)

        regr = merge_dfs([regr,API_pivot])

        regr.drop(columns = ["coordinates","area","radius"],inplace=True)
        #saving pivot table in database for summary display
        regr = merge_dfs_out([rent,sales,ratio,regr])
        regr.to_sql("regression",con_sum,if_exists="replace")
        
        #regression on each
        rent = regr.drop(columns=["sales","roi"]).dropna()
        rent = regression(rent)
        rent[0].to_sql("rent0",con_sum,if_exists="replace",index=False)
        rent[1].to_sql("rent1",con_sum,if_exists="replace",index=True)

        sales = regr.drop(columns=["rent","roi"]).dropna()
        sales = regression(sales)
        sales[0].to_sql("sales0",con_sum,if_exists="replace",index=False)
        sales[1].to_sql("sales1",con_sum,if_exists="replace",index=True)

        ratio = regr.drop(columns=["rent","sales"]).dropna()
        ratio = regression(ratio)
        ratio[0].to_sql("ratio0",con_sum,if_exists="replace",index=False)
        ratio[1].to_sql("ratio1",con_sum,if_exists="replace",index=True)
        
        #clickable link to summary
        df = pd.DataFrame({"SUMMARY":[f"/demo_summary"]})
        df["SUMMARY"] = df["SUMMARY"].apply(lambda x: '<a href="{0}">Click to view table only summary(For saving)</a>'.format(x))
        
        
        search = pd.read_sql("search",con_search)
        
        return (
            df.to_html(escape=False)+render_template("n.html")+
            search.to_html()+render_template("n.html")+
#             "GEO PLOTTING PLACES OF INTEREST"+render_template("n.html")+
            #geo_plt+
#             "RESULT OF GOOGLE GEO API SCRAPING"+
#             api.to_html(escape=False)+
            "SHOWING RESULTS FOR INPUT:" +render_template("n.html")+
            
            "RENT: $/SQFT"+render_template("n.html")+
            rent_plt+render_template("n.html")+
            "REGRESSION ANALYSIS ON IMPACT OF FACTORS REGARDING RENT"+render_template("n.html")+
            rent[0].to_html()+render_template("n.html")+
            rent[1].to_html()+render_template("n.html")+
            "SALES: $/SQFT"+render_template("n.html")+
            sales_plt+render_template("n.html")+
            "REGRESSION ANALYSIS ON IMPACT OF FACTORS REGARDING SALES"+render_template("n.html")+
            sales[0].to_html()+render_template("n.html")+
            sales[1].to_html()+render_template("n.html")+
            "ROI (PER YERAR: ROI = RENT*12/SALES*100)"+render_template("n.html")+
            ratio_plt+render_template("n.html")+
            "REGRESSION ANALYSIS ON IMPACT OF FACTORS REGARDING ROI"+render_template("n.html")+
            ratio[0].to_html()+render_template("n.html")+
            ratio[1].to_html()+render_template("n.html")+
            df.to_html(escape=False)
        )
        
        
    return render_template("us_local.html")

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
    regr = pd.read_sql("regression",con)
    
    con = create_engine("sqlite:///search.sqlite")
    search = pd.read_sql("search",con)
#     n = pd.DataFrame().to_html()
    return(
            
        "Result for:"+render_template("n.html")+
        search.to_html()+render_template("n.html")+
        "Regression analysis on rent"+render_template("n.html")+
        rent0.to_html()+rent1.to_html()+render_template("n.html")+
        "Regression analysis on sales"+render_template("n.html")+
        sales0.to_html()+sales1.to_html()+render_template("n.html")+
        "Regression analysis on ROI"+render_template("n.html")+
        ratio0.to_html()+ratio1.to_html()+render_template("n.html")+
        "Google API results"+render_template("n.html")+
        api.to_html(escape=False)+render_template("n.html")+
        "Pivot table of crime rates, population density, count and mean of POIS"+render_template("n.html")+
        regr.to_html()
        
        
    )

@app.route("/demo", methods=["GET", "POST"])
def demo():
    if request.method == "POST":
        name = request.form["city"]
        name = name.lower()
        name = name.replace(" ","_")
        con = create_engine("sqlite:///input.sqlite")
        inp = pd.DataFrame([name])
        inp.to_sql("input",con,if_exists="replace",index=False)
        return(
            "GEO plot of place of interests:" + render_template("n.html")+
            render_template(f"demo/{name}.html") + render_template("n.html")+
            render_template(f"demo/{name}_demo.html")

        )
    return render_template("demo.html")

@app.route("/demo_summary")
def check():
    con = create_engine("sqlite:///input.sqlite")
    inp = pd.read_sql("input",con).values[0][0]
    
    return render_template(f"demo/{inp}_summary.html")

if __name__ == "__main__":
    app.run(debug=True)