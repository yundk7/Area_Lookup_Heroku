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
import json
import math
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

def plotly_geo(df):
    df["size"] = 10
    df.loc[df['poi'] == "YOU ARE HERE", 'size'] = 20
    df.loc[df['poi'] == "YOU ARE HERE", 'reviews'] = 0
    df["reviews"].fillna(0,inplace=True)
    
    ptoken = key("ptoken")
    px.set_mapbox_access_token(ptoken)
    hover = ["reviews"]
#     if hover in (df.columns):
#         hover = ["name","reviews"]
#     else:
#         hover = []
    fig = px.scatter_mapbox(df, lat="Y", lon="X", color = "poi", size = "size", hover_name="name",zoom = 13, hover_data = hover)
    fig.update_layout(autosize=True,width=1500,height=750)
        #                           ,margin=go.layout.Margin(l=50,r=50,b=100,t=100,pad=4))
    return(py.offline.plot(fig,output_type="div"))

def google_geo(srch_list,pois,radius):
    records = pd.DataFrame()
    gkey = key("gkey")
    pois = pois.split(",")
    for s in srch_list:
        target_url = (f'https://maps.googleapis.com/maps/api/geocode/json?address={s}&key={gkey}')
        geo_data = requests.get(target_url).json()
        target_adr = geo_data["results"][0]["formatted_address"]
        lat = geo_data["results"][0]["geometry"]["location"]["lat"]
        lng = geo_data["results"][0]["geometry"]["location"]["lng"]
        target_coordinates = str(lat) + "," + str(lng)
        link = geo_data["results"][0]["place_id"]
        center_df = pd.DataFrame({"center":[s],"poi":["YOU ARE HERE"],"name":["YOU ARE HERE"],"address":[target_adr],"link":[link],"Y":[float(lat)],"X":[float(lng)]})
        records = records.append(center_df)
        for poi in pois:
            params = {
                "location": target_coordinates,
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
                    price = "NA"
                try:
                    link=places_data["results"][int(n)]["place_id"]
                except KeyError:
                    link = "NA"
                try:
                    score = places_data["results"][int(n)]["rating"]
                except KeyError:
                    score = "NA"
                try:
                    reviews = int(places_data["results"][int(n)]["user_ratings_total"])
                except KeyError:
                    reviews = "NA"
                try:
                    lat1 = places_data["results"][int(n)]["geometry"]["location"]["lat"]
                except KeyError:
                    lat1 = "NA"
                try:
                    lng1 = places_data["results"][int(n)]["geometry"]["location"]["lng"]
                except KeyError:
                    lng1 = "NA"
                content = pd.DataFrame ({"center":target_adr,"poi":poi,
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
    records = records[["center","poi","name","score","reviews","price","price","link","address","X","Y"]]
    records["link"]=records["link"].apply(lambda x: '<a href="https://www.google.com/maps/place/?q=place_id:{0}">link</a>'.format(x))
    return(records)

def kakao_api(centers_inp,pois_inp,radius):
    centers = centers_inp.split(",")
    pois = pois_inp.split(",")
    records = pd.DataFrame()
    for center in centers:
        url = 'https://dapi.kakao.com/v2/local/search/keyword.json?query='+center
        kkey = key("kkey")
        headers = {"Authorization": kkey}
        result = json.loads(str(requests.get(url,headers=headers).text))
        #     return result
        match_first = result['documents'][0]
        y = match_first['y']
        x = match_first['x']
        adr = match_first["address_name"]
        place_url = result["documents"][0]["place_url"]
        center_df = pd.DataFrame({"center":[center],"poi":["YOU ARE HERE"],"name":["YOU ARE HERE"],"address":[adr],"distance":[0],"link":[place_url],"X":[float(x)],"Y":[float(y)]})
        records = records.append(center_df)

        for poi in pois:
            page = 1
            size = 15
            last_page = 100
                # for query in queries:
            while page <= last_page:
                url = f"https://dapi.kakao.com/v2/local/search/keyword.json?y={y}&x={x}&radius={radius}&query="+poi+f"&page={page}"
                headers = {"Authorization": kkey}
                result1 = json.loads(str(requests.get(url,headers=headers).text))
                page+=1
                last_page = math.ceil(float(result1["meta"]["pageable_count"]/size))
                for n in range(0,len(result1["documents"])):
                    name = result1["documents"][n]["place_name"]
        #                name=str(name).split(" ")[0]
                    address = result1["documents"][n]["road_address_name"]
                    distance = result1["documents"][n]["distance"]
                    place_url = result1["documents"][n]["place_url"]
                    x1 = result1["documents"][n]["x"]
                    y1 = result1["documents"][n]["y"]
                    add = pd.DataFrame({"center":[center],"poi":[poi],
                                          "name":[name],
                                           "address":[address],
                                           "distance":[distance],
                                           "link":[place_url],
                                           "X":[float(x1)],
                                           "Y":[float(y1)]})
                    records=records.append(add)
    records.reset_index(drop=True,inplace=True)
    records["link"]=records["link"].apply(lambda x: '<a href="{0}">link</a>'.format(x))
    return (records)

#========================================================================


@app.route("/")
def home():
    #loading homepage = preparing database
    #for USA database
    con = create_engine("sqlite:///us_db.sqlite")
    #upload database if no table exist(when heroku is restarted)
    if len(con.table_names())==0:
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
        "/us",
        "/ggl",
        "/kakao"
    ]
    
    df["Content"] = [
        "Here at the home page, data is stored in temporary sqlite database",
        "With data scraped and gathered through Census bureau, Google API, Zillow Datasets, analyze local amenities and its impact on real estate value",
        "Searches and plots places of interest with respect to input location as center. Google API Used",
        "카카오 KAKAO rest API를 검색하여 관심지역을 검색, 맵핑합니다."
    ]
    df["Page"] = df["Page"].apply(lambda x: '<a href="{0}">{0}</a>'.format(x))
    
    return(df.to_html(escape=False))

@app.route("/us", methods=["GET", "POST"])
def us():
    if request.method == "POST":
        typ = request.form["typ"]
        srch = request.form["srch"]
        poi = request.form["pois"]
        typ = typ.lower()
        #to take account typing differences such as space and capital letters
        srch = srch.upper().replace(" ","").split(",")
        poi = poi.upper()
        
        #call in information to find what zipcodes are in search area(eg:chicago)
        con = create_engine("sqlite:///us_db.sqlite")
        city = pd.read_sql("city",con)
        city = city[city[typ].astype(str).str.replace(" ","").str.upper().isin(srch)]
        #from zip codes obtained, select information needed from each database
        zips = list(city["zip"])
        
#         rent = pd.read_csv("https://raw.githubusercontent.com/yundk7/area_lookup_heroku/master/local/Zip_ZriPerSqft_AllHomes.csv")
        rent = pd.read_sql("zillow_rent",con)
        rent = zillowELT(rent,zips)
        rent_plt = zillowplot(rent.T)
        
#         sales = pd.read_csv("https://raw.githubusercontent.com/yundk7/area_lookup_heroku/master/local/Zip_MedianListingPricePerSqft_AllHomes.csv")
        sales = pd.read_sql("zillow_sales",con)
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
        
        #since heroku is limited with request time, sampling out 10 zip codes to analyze
        sample = 10
        if len(df) < sample:
            sample = len(df)
        df = df.sample(sample)
        
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
        regr_rent[1].to_sql("rent1",con,if_exists="replace",index=True)
        
        sales_df = sales.iloc[:,-5:]
        sales_df = pd.DataFrame(sales_df.mean(axis=1))
        regr_sales = merge_dfs([sales_df,regr])
        regr_sales = regression(regr_sales)
        regr_sales[0].to_sql("sales0",con,if_exists="replace",index=False)
        regr_sales[1].to_sql("sales1",con,if_exists="replace",index=True)
        
        ratio_df = ratio.T.iloc[:,-5:]
        ratio_df = pd.DataFrame(ratio_df.mean(axis=1))
        regr_ratio = merge_dfs([ratio_df,regr])
        regr_ratio = regression(regr_ratio)
        regr_ratio[0].to_sql("ratio0",con,if_exists="replace",index=False)
        regr_ratio[1].to_sql("ratio1",con,if_exists="replace",index=True)
        
        #clickable link to summary
        df = pd.DataFrame({"SUMMARY":["/summary"]})
        df["SUMMARY"] = df["SUMMARY"].apply(lambda x: '<a href="{0}">Click to view table only summary(For saving)</a>'.format(x))
        
        return (
            "PLEASE NOTE THAT DUE TO REQUEST TIME LIMIT ONLINE, UP TO 10 ZIP CODES WERE SAMPLED FOR ANALYSIS!"+
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
        
        
    return render_template("us.html")

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
        "PLEASE NOTE THAT DUE TO REQUEST TIME LIMIT ONLINE, UP TO 10 ZIP CODES WERE SAMPLED FOR ANALYSIS!"+render_template("n.html")+    
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



if __name__ == "__main__":
    app.run(debug=True)