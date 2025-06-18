# Home Assignment

The aim of this assignment is to showcase candidates knowledge of cloud architecture and software engineering / programming skills. Therefore this assignment has two parts:

1. Describe high-level vision for the data platform and 
2. Write specific ETL script that should be run on a provided small set of dummy data.

The assignment should not take more than 3-4 hours total (it can take a lot less). The output should not be “final version”, but rather a strong first draft, that we could discuss on our review call. 

## Background
Eliq is a company that provides energy insights for our clients. The output can be in several forms, but underneath all client facing outputs is our API. We have two APIs exposed to clients - Data Management API and Insights API.

Data Management API is used to ingest energy consumption and other relavant data in our production databases. We use Cassandra and Azure SQL databases for this type of data. Energy consumption data can be of several different varieties:
1. Different fuels - electricity and gas;
2. Different “channels” - import, export, production and consumption;
3. Energy data can be of different frequency as well - from monthly to daily to hourly to 15 minute intervals. 

Another data type is a home profile data - information about the location where the energy is consumed. This data is stored in document database. We utilise a third party data sources in our analysis. The two most notable are - weather data and electricity market prices data.

Insights API returns energy insights utilising all the data that we have gathered. By querying this API with location ID our clients can get:
1. Forecasted energy consumption for location for different time horizons - from next 24 hours to next 12 months;
2. Energy usage categories - estimated energy consumption categories based on usage patterns and additional data like home profile and weather data;
3. Solar energy disaggregation - estimating how much solar energy location has produced given only their imported and exported energy;
4. Similar homes comparison - how does the location in question compared with other locations that have very similar characteristics;

## Part 1: Data Platform Architecture

Describe (and maybe draw a few boxes) how would you build a data platform for Eliq. What components would you use? How would you integrate with data sources? Who would be responsible for the different parts? How different parts of the company - data science, engineering, customer success, sales - could interact with this platform? 

Write down the first user stories and tasks. Tell us how you would go about building it if the plan would be approved?

### The list of data sources to consider:
1. Energy data - as described above, can be of different fuels (electricity, gas, district heating), comes from different “devices” (import, export, production, consumption) and can be in different granularity (from monthly data points to 15 minutes frequency data)
2. Home profile data - documents in document database with attributes for the specific location (what type of heating it has, how many people live in it, etc.)
3. Weather data - standard, row for each time period, data format. It is sampled for a fixed number of GPS coordinates in each country. Frequency - hourly.
4. Electricity price data - next day spot prices for each hour of the day.
5. Our internal services that generate insights on the fly and do not save anything in the databases.


You should not spend a lot of time here. I would expect a platform design that would be the basis for our discussion on the review call and that would contain all the major components, but not necessarily the exhaustive list of them.

## Part 2: ETL for handling difficult energy data formats

Write an ETL script to transform given data (you can find it in the data folder) into something that would be easier to query and analyse. The exact format is up to you to decide. The only requirement is - it has to be useful and meaningful for analytical queries. 

Data
Data is in parquet format. Schema description:

```
client_id: string
date: date32
ext_dev_ref: string
energy_consumption: list<item: int64>
resolution: string
```

Data contains a year worth of energy consumption data for a single meter (`ext_dev_ref` is unique identifier of the meter). Dates are in local time. You can assume that the location is in Europe/Vilnius time zone. Each energy consumption array contains 24 values of hourly energy consumption readings. The actual values does not matter for this exercise and therefore it’s just the same value for all hours.
