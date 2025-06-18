

import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from pathlib import Path
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

class EnergyETL:
    def __init__(self, input_file="home_assignment_raw_data.parquet", output_dir="output"):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.local_tz = pytz.timezone('Europe/Vilnius')
        self.utc_tz = pytz.UTC

    def extract(self):
        """Load and validate parquet data"""
        if not self.input_file.exists():
            raise FileNotFoundError(f"File not found: {self.input_file}")
        
        df = pd.read_parquet(self.input_file)
        print(f" Loaded {len(df):,} records")
        return df

    def transform(self, df):
        """Transform the arrays to hourly records """
                
        energy_df = pd.DataFrame(df['energy_consumption'].tolist(), index=df.index)
        energy_df.columns = [f'hour_{i}' for i in range(24)]
        
        # hourly format
        hourly = energy_df.reset_index().melt(
            id_vars='index', var_name='hour_str', value_name='energy_consumption_kwh'
        )
        
        # Adding metadata
        hourly['hour'] = hourly['hour_str'].str.extract(r'(\d+)').astype(int)
        hourly = hourly.merge(df.reset_index(), on='index').drop(columns=['index', 'hour_str'])
        hourly['date_dt'] = pd.to_datetime(hourly['date'])
        
        # Just a safety check for date_time format 
        try:
            hourly['timestamp_local'] = (
                hourly['date_dt'] + pd.to_timedelta(hourly['hour'], unit='h')
            ).dt.tz_localize(self.local_tz, ambiguous='NaT', nonexistent='shift_forward')
        except Exception:
            hourly['timestamp_local'] = hourly['date_dt'] + pd.to_timedelta(hourly['hour'], unit='h')
        
        if hourly['timestamp_local'].dt.tz is not None:
            hourly['timestamp_utc'] = hourly['timestamp_local'].dt.tz_convert(self.utc_tz)
        
        # maybe if no time zone specified converting to UTC #TODO
        else: 
            hourly['timestamp_utc'] = hourly['timestamp_local'] - pd.Timedelta(hours=2)
        
        # analytics features
        dt = hourly['timestamp_local'].dt
        hourly = hourly.assign(
            location_id='Europe/Vilnius',
            meter_id=hourly['ext_dev_ref'],
            year=dt.year,
            month=dt.month,
            day=dt.day,
            hour_of_day=dt.hour,
            day_of_week=dt.dayofweek,
            quarter=dt.quarter,
            is_weekend=(dt.dayofweek >= 5),

            # General Seasonal distrubution for sample analysis 
            season=dt.month.map({12: 'winter', 1: 'winter', 2: 'winter',
                                3: 'spring', 4: 'spring', 5: 'spring',
                                6: 'summer', 7: 'summer', 8: 'summer',
                                9: 'autumn', 10: 'autumn', 11: 'autumn'}),
            
            # Sample hte time periods for off_peak and Peak hours
            time_period=np.where((dt.hour >= 7) & (dt.hour <= 22), 'peak', 'off_peak'),
            load_period=np.select([
                dt.dayofweek >= 5,
                (dt.dayofweek < 5) & (dt.hour.isin([7, 8, 9, 18, 19, 20, 21])),
                (dt.dayofweek < 5) & (dt.hour >= 10) & (dt.hour <= 17),
                (dt.dayofweek < 5) & ((dt.hour >= 22) | (dt.hour <= 6))
            ], ['weekend', 'peak', 'shoulder', 'off_peak'], default='other'),
            energy_consumption_kwh=pd.to_numeric(hourly['energy_consumption_kwh'], errors='coerce'),
            
            # Flagging for data quality
            data_quality_flag=np.where(
                pd.to_numeric(hourly['energy_consumption_kwh'], errors='coerce').isnull(), 'missing',
                np.where(pd.to_numeric(hourly['energy_consumption_kwh'], errors='coerce') < 0, 'negative',
                        np.where(pd.to_numeric(hourly['energy_consumption_kwh'], errors='coerce') == 0, 'zero', 'valid'))
            )
        )
        
        
        daily_stats = hourly.groupby(['location_id', 'meter_id', 'date'])['energy_consumption_kwh'].agg([
            ('daily_total_kwh', 'sum'), ('daily_avg_kwh', 'mean')
        ]).reset_index()
        
        hourly = hourly.merge(daily_stats, on=['location_id', 'meter_id', 'date'])
        hourly['consumption_vs_daily_avg'] = (hourly['energy_consumption_kwh'] / hourly['daily_avg_kwh']).round(3)
        
        print(f"Created {len(hourly):,} hourly records")
        return hourly

    def load(self, df):
                
        
        main_path = self.output_dir / "energy_data.parquet"
        df.to_parquet(main_path, index=False)
        
        # Output to different formats depending on the pipe line CSV Chosen as an example 
        sample_path = self.output_dir / "energy_sample.csv"
        df.head(1000).to_csv(sample_path, index=False)
        
       
        daily = df.groupby(['location_id', 'date', 'season', 'is_weekend']).agg({
            'energy_consumption_kwh': ['sum', 'mean', 'max', 'min'],
            'daily_total_kwh': 'first'
        }).round(2)
        daily.columns = ['_'.join(col) for col in daily.columns]
        daily.reset_index().to_parquet(self.output_dir / "daily_summary.parquet", index=False)
        
        # Creating a report in JSON
        report = {
            'timestamp': datetime.now().isoformat(),
            'records_processed': len(df),
            'date_range': {'start': str(df['date'].min()), 'end': str(df['date'].max())},
            'total_consumption_kwh': float(df['energy_consumption_kwh'].sum()),
            'avg_hourly_kwh': float(df['energy_consumption_kwh'].mean()),
            'data_quality': df['data_quality_flag'].value_counts().to_dict(),
            'files_created': ['energy_data.parquet', 'energy_sample.csv', 'daily_summary.parquet']
        }
        
        with open(self.output_dir / "report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Files saved to: {self.output_dir}")
        return report

    def run(self):
        
        start_time = datetime.now()
        df_raw = self.extract()
        df_transformed = self.transform(df_raw)
        report = self.load(df_transformed)
        
        duration = datetime.now() - start_time
        print(f"\nETL completed in {duration}")
        print(f"Processed {report['records_processed']:,} records")
        print(f"Total consumption: {report['total_consumption_kwh']:,.1f} kWh")
        print(f"Average hourly: {report['avg_hourly_kwh']:.2f} kWh")
        
        return df_transformed, report

# Functions for streamlit app just for visualiasation
def load_processed_data(output_dir="output"):
    
    try:
        files = ['energy_data.parquet', 'energy_sample.csv']
        for file in files:
            path = Path(output_dir) / file
            if path.exists():
                if file.endswith('.parquet'):
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path)
                
                
                for col in ['timestamp_local', 'timestamp_utc', 'date']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                return df
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def load_report(output_dir="output"):
    
    try:
        path = Path(output_dir) / "report.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error loading report: {e}")
        return None


# generating demo data just incase the app dosen't function 
def generate_demo_data(days=30):
    
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=days)
    
    data = []
    for date in dates:
        for hour in range(24):
            consumption = max(0, 50 + 30 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 8))
            if date.weekday() >= 5:
                consumption *= 0.8
                
            data.append({
                'location_id': 'Europe/Vilnius',
                'client_id': '111111',
                'meter_id': 'ext_device_1',
                'date': date.date(),
                'hour_of_day': hour,
                'timestamp_local': date.replace(hour=hour),
                'energy_consumption_kwh': round(consumption, 2),
                'season': ['winter', 'winter', 'spring', 'spring', 'spring', 'summer', 
                          'summer', 'summer', 'autumn', 'autumn', 'autumn', 'winter'][date.month-1],
                'time_period': 'peak' if 7 <= hour <= 22 else 'off_peak',
                'is_weekend': date.weekday() >= 5,
                'data_quality_flag': 'valid'
            })
    
    df = pd.DataFrame(data)
    daily_totals = df.groupby('date')['energy_consumption_kwh'].sum()
    df['daily_total_kwh'] = df['date'].map(daily_totals)
    return df

def main():
    parser = argparse.ArgumentParser(description="Eliq Energy ETL Pipeline")
    parser.add_argument('input', nargs='?', default='home_assignment_raw_data.parquet')
    parser.add_argument('output', nargs='?', default='output')
    args = parser.parse_args()
    
    
    input_paths = [args.input, f"data/{args.input}", "data/home_assignment_raw_data.parquet"]
    input_file = None
    
    for path in input_paths:
        if Path(path).exists():
            input_file = path
            break
    
    if not input_file:
        print(f"Input file not found. Tried: {input_paths}")
        return 1
    
    try:
        etl = EnergyETL(input_file, args.output)
        etl.run()
        print("Can be visualised in and Ready for Streamlit!")
        return 0
    except Exception as e:
        print(f"ETL failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
