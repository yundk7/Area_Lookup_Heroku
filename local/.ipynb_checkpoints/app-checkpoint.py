from flask import (
    Flask,
    render_template,
    jsonify,
    request)
from IPython.display import HTML
import pandas as pd

import plotly as py
import plotly.graph_objs as go
import plotly.express as px
py.offline.init_notebook_mode(connected = True)

import requests
import json
import math
from bs4 import BeautifulSoup as bs
import re


app = Flask(__name__)

#FUNCTIONS=======================================================
def findzip(srch):
    srch = srch.split(",")
    zips = []
    for s in srch:
        gkey = "AIzaSyA-Rjp6nOeJp6815Xt1Kkuxc5XKMiKl_yA"
        target_url = (f'https://maps.googleapis.com/maps/api/geocode/json?address={s}&key={gkey}')
        geo_data = requests.get(target_url).json()
#         zips.append(geo_data["results"][0]["address_components"][-1]["long_name"])
        zips.append(geo_data["results"][0]["formatted_address"].split(", USA")[0][-5:])
    seperator = ","
    zips = seperator.join(zips)
    return(zips)

#to manipulate zillow data
def zillowELT(df_inp,typ_inp,srch_inp):
    df = df_inp.rename(columns = {"RegionName":"Zip"})
    typ = typ_inp
    srch = srch_inp.split(",")
    srch = [x.upper() for x in srch]
    srch = [x.replace(" ","") for x in srch]
    df[typ] = df[typ].astype(str).str.upper().str.replace(" ","")
    df = df[df[typ].isin(srch)]
    df["name"] = df[typ].astype(str) + " : " + df["Zip"].astype(str)
    df = df.groupby("name").mean().T
    #converting Jan-10 to datetime as 2010-01-01
    df.index = pd.to_datetime(df.index,format = "%b-%y",errors = "coerce")
    df = df[df.index.isnull()==False]
    return(df)

#returns html code of plot
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
    return(py.offline.plot(fig,output_type = "div"))

#to webscrape crime rates for each zipcodes
def crimerate(zips_inp):
    df = pd.DataFrame()
    if zips_inp[-1] == ",":
        zips_inp = zips_inp[:-1]
    zips = zips_inp.split(",")
    for z in zips:
        url = f'https://www.bestplaces.net/crime/zip-code/state/city/{z}'
        response = requests.get(url)

        # Create a Beautiful Soup object
        soup = bs(response.text, 'html.parser')

        # Print all divs with col-md-12
        divs = soup.find_all("div", {"class": "col-md-12"})
        try:
            s = str(divs[1]).split("violent crime is ")[1]
            result = re.findall(r"[-+]?\d*\.\d+|\d+", s)
        except IndexError:
            result = ["NAN"]
        vcr = result[0]
        try:
            s = str(divs[1]).split("property crime is ")[1]
            result = re.findall(r"[-+]?\d*\.\d+|\d+", s)
        except IndexError:
            result = ["NAN"]
        pcr = result[0]
        apnd = pd.DataFrame({"zip":[z],
                            "violent crime": [vcr],
                            "property crime": [pcr]})
        df = df.append(apnd)
    return(df)

def google_geo(srch_list,pois,radius):
    records = pd.DataFrame()
    gkey = "AIzaSyA-Rjp6nOeJp6815Xt1Kkuxc5XKMiKl_yA"
#     srch_list = ["91765","60607"]
    # srch_list = ["walnut high school","235 west van buren"]
#     pois = "restaurants,subway station"
#     radius = 15000
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
#         return(records.to_html())
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
#     records["link"]=records["link"].apply(lambda x: '<a href="https://www.google.com/maps/place/?q=place_id:{0}">link</a>'.format(x))
    return(records)

def kakao_api(centers_inp,pois_inp,radius):
    centers = centers_inp.split(",")
    pois = pois_inp.split(",")
    records = pd.DataFrame()
    for center in centers:
        url = 'https://dapi.kakao.com/v2/local/search/keyword.json?query='+center
        headers = {"Authorization": "KakaoAK 8809fcb48aa9900788adbd9f162c6b25"}
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
                headers = {"Authorization": "KakaoAK 8809fcb48aa9900788adbd9f162c6b25"}
                result1 = json.loads(str(requests.get(url,headers=headers).text))
                page+=1
                last_page = int(result1["meta"]["pageable_count"]/size)+1
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
#                     records["X"] = pd.to_numeric(records["X"],errors = "coerce")
#                     records["Y"] = pd.to_numeric(records["Y"],errors = "coerce")
                    records=records.append(add)
#     records = records[["center","poi","name","address","distance","link","X","Y"]]
#     records.reset_index(drop = True,inplace = True)
#     records["link"]=records["link"].apply(lambda x: '<a href="http://place.map.kakao.com/{0}">link</a>'.format(x))
#     records["link"]=records["link"].apply(lambda x: '<a href="{0}">link</a>'.format(x))
    return (records)

def plotly_geo(df):
    df["size"] = 10
    df.loc[df['poi'] == "YOU ARE HERE", 'size'] = 20
    px.set_mapbox_access_token("pk.eyJ1IjoidGl2bWU3IiwiYSI6ImNrMWEwZDVtNDI4Zm4zYm1vY3o3Z25zejEifQ._yTPkj3nXTzor72zIevLCQ")
    hover = "reviews"
    if hover in (df.columns):
        hover = ["reviews"]
    else:
        hover = []
    fig = px.scatter_mapbox(df, lat="Y", lon="X", color = "poi", size = "size", hover_name="name",zoom = 13, hover_data = hover)
    fig.update_layout(autosize=True,width=1500,height=750)
        #                           ,margin=go.layout.Margin(l=50,r=50,b=100,t=100,pad=4))
    return(py.offline.plot(fig,output_type="div"))
#========================================================================


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        typ = request.form["typ"]
        typ1 = typ
        srch = request.form["srch"]
        radius = request.form["radius"]
        pois = request.form["pois"]
        
        if typ == "Address":
            addresses = srch.split(",")
            srch = findzip(srch)
            if radius == "":
                radius = 15000
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
                API_record = google_geo(zips,pois,radius)
            if typ1 == "Address":
                API_record = google_geo(addresses,pois,radius)
            geo_plot = plotly_geo(API_record)
        else:
            API_record = pd.DataFrame()
            geo_plot = ""
        return(crime_html
               +"RENT: $/sqft"
               +rent_html
               +"SALES: $/sqft"
               +sales_html
               +"RATIO (FOR SIMPLE CALCULATION, USED EQATION: ROI = RENT*12/SALES*100)"
               +ratio_html
               +API_record.to_html()
               +geo_plot
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