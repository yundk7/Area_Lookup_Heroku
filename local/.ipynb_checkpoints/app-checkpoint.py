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
                    reviews = places_data["results"][int(n)]["user_ratings_total"]
                except KeyError:
                    reviews = "NA"
                content = pd.DataFrame ({"depart":target_adr,"poi":poi,
                                    "name":[places_data["results"][int(n)]["name"]],
                                    "score":score,
                                     "reviews":reviews,
                                     "price":price,
                                     "link":link,
                                    "address":[places_data["results"][int(n)]["vicinity"]],
            #                        "distance":distance,
            #                         "drive":duration,
            #                         "public":transit_dur,
            #                         "walk":walk_dur
                                        })
                records = records.append(content)
                n+=1
    records.reset_index(drop = True,inplace = True)
    records["link"]=records["link"].apply(lambda x: '<a href="https://www.google.com/maps/place/?q=place_id:{0}">link</a>'.format(x))
    return(records.to_html(escape=False))

# def kakao_api(centers_inp,pois_inp,radius):
#     centers = centers_inp.split(",")
#     pois = pois_inp.split(",")
    
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
        else:
            API_record = ""
        return(crime_html
               +"RENT: $/sqft"
               +rent_html
               +"SALES: $/sqft"
               +sales_html
               +"RATIO (FOR SIMPLE CALCULATION, USED EQATION: ROI = RENT*12/SALES*100)"
               +ratio_html
               +API_record
              )
    
    return render_template("form.html")

# @app.route("/kr", methods=["GET", "POST"])
# def kr():
#     if request.method == "POST":
#         asdf
    
#     return render_template("form_kr.html")
if __name__ == "__main__":
    app.run(debug=True)