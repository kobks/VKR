import difflib
import math
import random
import time
import urllib.request

import keras as keras
from PIL import Image

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import requests
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.scrollview import ScrollView

from kivy.core.window import Window
import dfgui

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics

# Window.size = (720, 1280)
Window.size = (480, 720)

weight = 80
user_program = []
cals_limit = 3000
exercises = {}

calories1 = pd.read_excel("cal_table.xlsx")
calories1.drop("Наименование", axis=1)
calories1.rename(columns={'Name': 'food',
                          'Protein': 'Protein, g',
                          'Fat': 'Fat, g',
                          'Carbs': 'Carbs, g',
                          'Calories': 'Calories, kcal'}, inplace=True)
calories3 = pd.read_excel("foods.xlsx")
calories = pd.concat([calories1, calories3])
calories.drop("Наименование", axis=1)
calories.reset_index(drop=True)
print(calories1)

from kivy.config import Config
Config.set('kivy', 'keyboard_mode', 'systemanddock')

def get_ingridients(m):
    nitro = str(10 * m / 1000)
    salt = str(15 * m / 1000)
    start = str(0.5 * m / 1000)
    dextrose = str(5 * m / 1000)
    salting_time = str(round(m / 500 * 2))

    return {
            'nitro': nitro,
            'salt': salt,
            'start': start,
            'dextrose': dextrose,
            'salting_time': salting_time
            }
class ContainerApp(GridLayout):
    def calculate(self):
        try:
            mass = int(self.text_input.text)
        except:
            mass = 0

        ingr = get_ingridients(mass)
        self.salt.text = ingr.get("salt") + ' + 5'
        self.nitro.text = ingr.get("nitro")
        self.dextrose.text = ingr.get("dextrose")
        self.start.text = ingr.get("start")
        self.salting_time.text = ingr.get("salting_time")

def get_all_coordinates(x):
    gym_coordinates = []
    for item in x.json()["features"]:
        gym_coordinates.append(item["geometry"]["coordinates"])
    return gym_coordinates


def points_create(l):
    points = []
    for item in l:
        points.append(f"{item[0]},{item[1]},pm2wtm{l.index(item) + 1}")
    return "~".join(points)

class HeartRateScreen(Screen):
    def build_and_compile_model(self, norm):
        model = keras.Sequential([
            norm,
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(loss='mean_absolute_error',
                      optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    def get_history(self):
        apptoken = "RQVBQEJyQktGXip6SltGImp2ej48BAAEAAAAAIjiO3VW3QImkgtdJCi7ahfpM592a_DZz09ZaMPA1HhlfA2Hyfgr5qKetRkSKDqF6_2x4abY7bx7wATpFlQdebB0bfCXq9BGQv3JDPVF1TTTSSGxQoEITJgIf6Uxq5wKVWf95EcCrP4oEmUbQkF0aGZPmkC6dC5iHrQGWctKe8004jQjGWA5jmKz8qyMSYSYz"

        r = requests.get('https://api-mifit-de2.huami.com/v1/sport/run/history.json', headers={
            'apptoken': apptoken
        }, params={
            'source': 'run.mifit.huami.com',
        })
        r.raise_for_status()
        return r.json()

    def calc_heartrate(self, model):
        ex = self.get_history()["data"]["summary"][-1]
        dist = float(ex["dis"]) / 1000
        speed = (float(ex["dis"]) / 1000) / (float(ex["run_time"]) / 3600)
        calories = float(ex["calorie"])
        time = float(ex["run_time"])
        heart_rate_real = float(ex["avg_heart_rate"])
        ans = ""
        # ans += f"Dist: {dist}, speed: {speed}, cal: {calories} "
        ans += f"Actual H/R: {heart_rate_real} "

        # total_running_time	total_calories	distance	avg_speed
        heart_rate_calculated = model.predict([time, calories, dist, speed])
        ans += f"Calculated H/R: {heart_rate_calculated[0][0]}\n"
        ans += "Based on the last training session\n"

        if heart_rate_calculated - heart_rate_real > 5:
            ans += "\nYour heart rate is low"
        elif heart_rate_real - heart_rate_calculated > 5:
            ans += "\nYour heartrate is high!"
        else:
            ans += "\nYour heartrate is normal"
        return ans
    def anal_hr(self):
        hist = self.get_history()

        running_heartrate = pd.read_csv("running_summary.csv")
        running_heartrate = running_heartrate[
            ['heart_rate', 'total_running_time', 'total_calories', 'distance', 'avg_speed']].dropna()

        running_train_dataset = running_heartrate.sample(frac=0.8, random_state=0)
        running_test_dataset = running_heartrate.drop(running_train_dataset.index)

        running_train_features = running_train_dataset.copy()
        running_test_features = running_test_dataset.copy()

        running_train_labels = running_train_features.pop('heart_rate')
        running_test_labels = running_test_features.pop('heart_rate')

        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(running_train_features))
        dnn_model = self.build_and_compile_model(normalizer)

        res = self.calc_heartrate(dnn_model)
        prev_ex = ""
        cnt = 1
        for item in hist:
            ex = self.get_history()["data"]["summary"][cnt - 1]
            dist = float(ex["dis"]) / 1000
            speed = (float(ex["dis"]) / 1000) / (float(ex["run_time"]) / 3600)
            calories = float(ex["calorie"])
            time = float(ex["run_time"])
            heart_rate_real = float(ex["avg_heart_rate"])
            prev_ex += f"Training session: {cnt}, Distance: {dist}, Average speed: {round(speed, 2)}\n"
            prev_ex += f"Calories spent: {calories}, Time: {time}\n"
            prev_ex += f"Average heart rate: {heart_rate_real}\n\n"
            cnt += 1
        prev_ex += res
        print(prev_ex)
        self.hr_data.text = prev_ex

class GraphsScreen(Screen):

    def plot_reps(self, history, exercise):
        history = history.astype(
            {"Training_number": int, f"{exercise}, reps": float, f"{exercise}, weight (kg)": float})

        sns.lmplot(x=f"Training_number", y=f"{exercise}, reps", data=history, fit_reg=True)
        # sns.lmplot(x=f"Training_number", y=f"{exercise}, weight (kg)", data=history, fit_reg=True)

        df_temp = history[["Date", f"{exercise}, reps", f"{exercise}, weight (kg)"]].copy()
        df_temp["dates"] = pd.to_datetime(history['Date']).apply(lambda date: date.toordinal())
        ax = sns.regplot(
            data=df_temp,
            x="dates",
            y=f"{exercise}, reps",
        )
        # Tighten up the axes for prettiness
        ax.set_xlim(df_temp['dates'].min() - 1, df_temp['dates'].max() + 1)
        ax.set_ylim(0, df_temp[f"{exercise}, reps"].max() + 1)
        # Replace the ordinal X-axis labels with nice, readable dates
        ax.set_xlabel('Date')
        new_labels = [datetime.date.fromordinal(int(item)) for item in ax.get_xticks()]
        ax.set_xticklabels(new_labels)
        ax.tick_params(axis='x', rotation=90)
        plt.savefig("reps_graph.png")

    def plot_weights(self, history, exercise):
        history = history.astype(
            {"Training_number": int, f"{exercise}, reps": float, f"{exercise}, weight (kg)": float})

        sns.lmplot(x=f"Training_number", y=f"{exercise}, weight (kg)", data=history, fit_reg=True)

        df_temp = history[["Date", f"{exercise}, reps", f"{exercise}, weight (kg)"]].copy()
        df_temp["dates"] = pd.to_datetime(history['Date']).apply(lambda date: date.toordinal())
        ax = sns.regplot(
            data=df_temp,
            x="dates",
            y=f"{exercise}, weight (kg)",
        )
        # Tighten up the axes for prettiness
        ax.set_xlim(df_temp['dates'].min() - 1, df_temp['dates'].max() + 1)
        ax.set_ylim(0, df_temp[f"{exercise}, weight (kg)"].max() + 1)
        # Replace the ordinal X-axis labels with nice, readable dates
        ax.set_xlabel('Date')
        new_labels = [datetime.date.fromordinal(int(item)) for item in ax.get_xticks()]
        ax.set_xticklabels(new_labels)
        ax.tick_params(axis='x', rotation=90)
        plt.savefig("weight_graph.png")
    def create_graphs(self):

        dummy_exercise = user_program.columns[2].split(",")[0]
        self.plot_weights(user_program, dummy_exercise)
        self.w_graph.source = "weight_graph.png"

        self.plot_reps(user_program, dummy_exercise)
        self.r_graph.source = "reps_graph.png"



class CalScreen(Screen):
    def change_limit(self):
        global cals_limit
        cal_lim = int(self.text_input_cal_limit.text)
        cals_limit = cal_lim
        self.text_cal_limit.text = f"Current calories limit: {cal_lim}"

    def get_calories(self, food):
        print("FOOD: ", food)
        print(calories["food"])

        foods = difflib.get_close_matches(food, calories["food"], n=10, cutoff=0.45)
        if foods:
            food_cal = calories[calories["food"] == foods[0]]
            print(food_cal.values.tolist())
            return food_cal.values.tolist()[0][-1]
        print(f"Food {food} not found!")
        return None

    def recommend_exercise(self, calories_proficit, weight, gym_activities):
        # calories are per 30 minutes
        ans = ""
        ans += f"You are {calories_proficit} kcal overlimit\n"
        suggestions = gym_activities[gym_activities["155-pound person"] <= calories_proficit]
        suggestions = suggestions.sample(len(suggestions) // 3).sort_values(by=['155-pound person'], ascending=False)
        recommendation = "The following activites can be suggested:\n"
        for item in suggestions.iterrows():
            cal_coef = item[1][1] + (item[1][2] - item[1][1]) / (155 - 125) * abs(155 - weight * 2.205)

            recommendation += f'{item[1][0]} for {int(calories_proficit / (cal_coef / 30))} minutes\n'
        print(recommendation)
        ans += recommendation
        return ans
    def calculate_calories(self):
        gym_activities = pd.read_excel("gym_activities.xlsx")  # calories burnt per 30 minutes

        print("TEST")
        text = self.text_input_foods.text
        foods = text.split("\n")
        # cals = 0
        calories_today = 0
        for food in foods:
            calories_today += self.get_calories(food)
        print("CALS: ", calories_today)
        s = ""

        s += f"Total {calories_today} kcal eaten today\n"

        if calories_today > cals_limit:
            s += f"You are over limit on daily calories, recommend an exercise\n"
            s += self.recommend_exercise(calories_today - cals_limit, weight, gym_activities)

        elif calories_today < cals_limit:
            s += f"Some foods you can eat today:\n"
            food_vars = calories[calories["Calories, kcal"] <= (cals_limit - calories_today)]
            # print(type(food_vars))
            for index, row in food_vars.iterrows():
                s += f"{row['food']}, kilo calories - {row['Calories, kcal']}\n"
                if index >= 10:
                    break
        self.cals_result.text = s

class HistoryScreen(Screen):
    def show_history(self):

        labels = []
        i = 0
        for item in user_program.columns:
            print(item)


            # print(user_program[item])
            col = f"{item}\n" + "\n".join(list(map(str, user_program[item]))) + 20*"\n"
            print(col)
            labels.append(col)
            i += 1

        self.history_output_1.text = labels[0]
        self.history_output_2.text = labels[1]
        self.history_output_3.text = labels[2]
        self.history_output_4.text = labels[3]
        self.history_output_5.text = labels[4]

        self.history_output_6.text = labels[5]
        self.history_output_7.text = labels[6]
        self.history_output_8.text = labels[7]
        self.history_output_9.text = labels[8]
        self.history_output_10.text = labels[9]

        self.history_output_11.text = labels[10]
        self.history_output_12.text = labels[11]
        self.history_output_13.text = labels[12]
        self.history_output_14.text = labels[13]
        self.history_output_15.text = labels[14]

        self.history_output_16.text = labels[15]
        self.history_output_17.text = labels[16]
        self.history_output_18.text = labels[17]
        self.history_output_19.text = labels[18]
        self.history_output_20.text = labels[19]

        self.history_output_21.text = labels[20]
        self.history_output_22.text = labels[21]
        self.history_output_23.text = labels[22]
        self.history_output_24.text = labels[23]
        self.history_output_25.text = labels[24]

        self.history_output_26.text = labels[25]
        self.history_output_27.text = labels[26]
        self.history_output_28.text = labels[27]
        self.history_output_29.text = labels[28]
        self.history_output_30.text = labels[29]

        self.history_output_31.text = labels[30]
        self.history_output_32.text = labels[31]
        self.history_output_33.text = labels[32]
        self.history_output_34.text = labels[33]
        self.history_output_35.text = labels[35]

    def my_output(self, X, weights, biases):
        return tf.add(tf.multiply(X, weights), biases)

    def loss_func(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))
    def get_recommended_weights(self, tr_history, ex_name):
        x_vals = tr_history[f"{ex_name}, reps"].astype(np.float32).to_numpy()
        y_vals = tr_history[f"{ex_name}, weight (kg)"].astype(np.float32).to_numpy()
        norm = np.linalg.norm(x_vals)  # To find the norm of the array
        normalized_array = x_vals / norm  # Formula used to perform array normalization

        my_opt = tf.optimizers.SGD(learning_rate=0.02)
        tf.random.set_seed(1)
        np.random.seed(0)
        weights = tf.Variable(tf.random.normal(shape=[1]))
        biases = tf.Variable(tf.random.normal(shape=[1]))
        history = list()

        for i in range(len(x_vals)):
            rand_index = np.random.choice(len(x_vals))
            rand_x = [normalized_array[rand_index]]
            rand_y = [y_vals[rand_index]]
            with tf.GradientTape() as tape:
                predictions = self.my_output(rand_x, weights, biases)
                loss = self.loss_func(rand_y, predictions)
            history.append(loss.numpy())
            gradients = tape.gradient(loss, [weights, biases])
            my_opt.apply_gradients(zip(gradients, [weights, biases]))
        return weights, biases

    def get_predicted_weight(self, history, model, w, b, weights, reps):
        pred_w = round(float(model(float(reps), w, b)), 2) / 2
        res = ""
        res += f"Predicted weights: {pred_w}, actual weights: {weights}\n"
        if pred_w < weights:
            res += "Good weights, recommend increasing reps"
        else:
            res += "Recommend increasing weights"
        return res
    def add_exec(self):
        # exercise, weight, reps, rest, t_num, date = datetime.datetime(2023, 5, 17)
        global user_program

        date = datetime.date(2023, 5, 17)
        exercise = self.text_input_tr_name.text
        weight = self.text_input_tr_weight.text
        reps = int(self.text_input_tr_reps.text)
        rest = int(self.text_input_tr_rest.text)
        t_num = int(self.text_input_tr_num.text)

        # exercise = "a"
        # weight = 10
        # reps = 10
        # rest = 10
        # t_num = 1

        ind = len(user_program)
        if user_program.index[user_program['Training_number'] == t_num].tolist():
            ind = user_program.index[user_program['Training_number'] == t_num].tolist()[0]

        user_program.loc[ind, ["Date"]] = date
        user_program.loc[ind, "Training_number"] = t_num
        user_program.loc[ind, f"{exercise}, reps"] = float(reps)
        user_program.loc[ind, f"{exercise}, weight (kg)"] = weight
        user_program.loc[ind, f"{exercise}, rest (s)"] = rest

        if len(user_program) > 10:
            w, b = self.get_recommended_weights(user_program, exercise)
            predict = self.get_predicted_weight(user_program, self.my_output, w, b, float(weight), reps)
            self.exec_recom.text = predict
class GPSScreen(Screen):
    flag_point = 0
    images = []
    closest_coords = []
    cur_map = 0

    def coord_distance(self, x1, y1, x2, y2):
        # print(f"{x1}, {y1}, {x2}, {y2}")
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def find_closes_gym(self, long, lat, gyms, dist, points):

        # global flag_point
        closest = 0
        min_dist = 100
        for j in gyms:
            d = self.coord_distance(long, lat, j[0], j[1])
            if d < dist:
                if d < min_dist:
                    min_dist = d
                    closest = points.index(j) + 1

        if closest:
            return f"Closest sports ground is {closest}, in {round(min_dist * 100000, 2)} meters!"

            if closest != self.flag_point:
                pass
                # notification_gym_nearby(closest, min_dist)
                flag_point = closest

        else:
            print("No closest point...")

    def get_coords(self):

        HSE_long = 37.64823890522192
        HSE_lat = 55.75362365163729

        long_start = 37.63912337086539
        lat_start = 55.7531586027803

        long_end = 37.63468163273671
        lat_end = 55.75671621603898

        ya_api = "be69a8a0-9ea0-42cf-ad7a-5b73199a308d"

        req_str = f"https://search-maps.yandex.ru/v1/?" \
                  "text=Спортивная площадка&" \
                  f"ll={long_start},{lat_start}&" \
                  "spn=0.552069,0.550552&" \
                  "type=biz&" \
                  "results=40&" \
                  "lang=ru_RU&" \
                  f"apikey={ya_api}"
        print(req_str)

        x = requests.get(req_str)
        gym_coordinates = get_all_coordinates(x)
        self.map_screenshot_gps.source = f"""https://static-maps.yandex.ru/1.x/?ll={long_start},{lat_start}&size=450,450&z=15&l=map&pt={long_start},{lat_start},pm2dol1~{points_create(gym_coordinates)}"""

        steps = 10
        dist = 0.01

        for i in range(steps):

            cur_lat = lat_start + (lat_end - lat_start) * i / steps
            cur_long = long_start + (long_end - long_start) * i / steps

            self.closest_coords.append( self.find_closes_gym(cur_long, cur_lat, gym_coordinates, dist, gym_coordinates) )
            self.images.append( f"""https://static-maps.yandex.ru/1.x/?ll={cur_long},{cur_lat}&size=450,450&z=15&l=map&pt={cur_long},{cur_lat},pm2dol1~{points_create(gym_coordinates)}""" )
            urllib.request.urlretrieve(self.images[i], f"map_{i}.png")

            self.map_screenshot_gps.source = f"map_0.png"
        self.cur_map = 0
        print(self.images)



    def update_map(self):
        print(self.closest_coords)

        if self.closest_coords:
            self.notif_label.font_size = "16sp"
            if self.cur_map <= len(self.images):
                self.map_screenshot_gps.source = f"map_{self.cur_map}.png"
                self.notif_label.text = self.closest_coords[self.cur_map]
                self.cur_map += 1


class MapScreen(Screen):
    def get_coords(self):
        HSE_long = 37.64823890522192
        HSE_lat = 55.75362365163729
        ya_api = "be69a8a0-9ea0-42cf-ad7a-5b73199a308d"

        req_str = f"https://search-maps.yandex.ru/v1/?" \
                  "text=Спортивная площадка&" \
                  f"ll={HSE_long},{HSE_lat}&" \
                  "spn=0.552069,0.550552&" \
                  "type=biz&" \
                  "results=15&" \
                  "lang=ru_RU&" \
                  f"apikey={ya_api}"
        # print(req_str)

        x = requests.get(req_str)
        gym_coordinates = get_all_coordinates(x)

        # HSE_map = requests.get(f"""https://static-maps.yandex.ru/1.x/?ll={HSE_long},{HSE_lat}&size=450,450&z=13&l=map&pt={points_create(gym_coordinates)}""")

        # print("TYPE: ", type(HSE_map))

        # "http://kivy.org/logos/kivy-logo-black-64.png"
        self.map_screenshot.source = f"""https://static-maps.yandex.ru/1.x/?ll={HSE_long},{HSE_lat}&size=450,450&z=13&l=map&pt={HSE_long},{HSE_lat},pm2dol1~{points_create(gym_coordinates)}"""


class InfoScreen(Screen):
    def calculate_BMI(self, weight, height):
        BMI = round(weight / (height / 100) ** 2, 1)
        s = ""

        if BMI <= 18.4:
            s += "You are underweight."
        elif BMI <= 24.9:
            s += "You are healthy."
        elif BMI <= 29.9:
            s += "You are over weight."
        elif BMI <= 34.9:
            s += "You are severely over weight."
        elif BMI <= 39.9:
            s += "You are obese."
        else:
            s += "You are severely obese."

        return f"{BMI}, " + s

    def calculate_bmi_cal(self):
        global cals_limit
        global weight
        try:
            age = int(self.text_input_age.text)
        except:
            age = 20

        try:
            height = int(self.text_input_height.text)
        except:
            height = 170

        try:
            weight = int(self.text_input_weight.text)
        except:
            weight = 80

        try:
            gender = int(self.text_input_gender.text)
            if gender != "M" or gender != "F":
                gender = "M"
        except:
            gender = "M"

        self.text_output_BMI.text = calculate_BMI(weight, height)

        cals = 3000 if gender == "M" else 2400
        self.text_output_calorie.text = str(cals)

class ProgramScreen(Screen):

    def dummy_data(self):
        global user_program

        dummy_exercise = user_program.columns[2].split(",")[0]
        print(dummy_exercise)

        date = datetime.date(2023, 5, 17)
        t_num = 1

        for i in range(100):
            a = round(random.uniform(5, 30), 0)
            b = round(random.uniform(10, 30), 0)

            new_row = {
                f'{dummy_exercise}, weight (kg)': a  + (i // 8),
               f'{dummy_exercise}, reps': b  + (i // 8),
               'Training_number': i,
               'Date': date  + datetime.timedelta(days= (i // 3))
                }
            user_program.loc[len(user_program.index)] = new_row
            new_row = {

            }

    def generate_program(self):
        global user_program
        global exercises
        levels = {'Intermediate': 2, 'Beginner': 1, 'Expert': 3}
        goals = {'Strength': 1, 'Plyometrics': 2, 'Cardio': 3, 'Stretching': 4, 'Powerlifting': 5,
                 'Strongman': 6, 'Olympic Weightlifting': 7}
        equipments = {'Bands': 1, 'Barbell': 2, 'Kettlebells': 3, 'Dumbbell': 4, 'Other': 5, 'Cable': 6,
                      'Machine': 7, 'Body Only': 8, 'Medicine Ball': 9, 'None': 10, 'Exercise Ball': 11,
                      'Foam Roll': 12, 'E-Z Curl Bar': 13}
        try:
            goal = goals.get(self.text_input_goals.text, default = 1)
        except:
            goal = 1

        try:
            equip = equipments.get(self.text_input_equip.text, default = 8)
        except:
            equip = 8

        try:
            level = levels.get(self.text_input_level.text, default = 1)
        except:
            level = 1

        print("INPUT: ", goal, equip, level)

        labels = ["mg_abdo", "mg_addu", "mg_abdu", "mg_bice",
                  "mg_calv", "mg_ches", "mg_fore", "mg_fore",
                  "mg_glut", "mg_hams", "mg_lats", "mg_lbac",
                  "mg_mbac", "mg_trap", "mg_neck", "mg_quad",
                  "mg_shou", "mg_tric"]



        df_gym = pd.read_csv("megaGymDataset.csv")
        feature_cols = ["Title", "BodyPart", "Equipment", "Level", "Rating", "RatingDesc"]
        X = df_gym[feature_cols]  # Features
        y = df_gym["Type"]  # Target variable
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        cats = ["Type", "Equipment", "Level", "RatingDesc"]

        df_gym_cat = df_gym

        for cat in cats:
            codes, uniques = pd.factorize(df_gym_cat[cat])
            df_gym_cat[cat] = codes
        df_gym_cat['Rating'] = df_gym_cat['Rating'].fillna(0)

        codes, uniques = pd.factorize(df_gym['Type'])

        # df_gym[df_gym["Type"] == "Cardio"]

        # df_gym.groupby('BodyPart').count()
        feature_cols = ["Type", "Equipment", "Level", "Rating", "RatingDesc"]

        parts = df_gym_cat['BodyPart'].unique()

        models = []
        partial_dfs = []

        for part in parts:
            temp_df = df_gym_cat[df_gym["BodyPart"] == part]
            temp_df.insert(0, 'New_ID', range(1, 1 + len(temp_df)))

            codes, uniques = pd.factorize(temp_df['Type'])
            temp_df['Type'] = codes

            # print(temp_df['Type'].astype('category').cat.categories , "\n")

            X = temp_df[feature_cols]  # Features
            y = temp_df["New_ID"]  # Target variable

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
            print(X_train)

            # Create Decision Tree classifer object
            clf_gym = DecisionTreeClassifier()

            # Train Decision Tree Classifer
            clf_gym = clf_gym.fit(X_train, y_train)

            # Predict the response for test dataset
            y_pred = clf_gym.predict(X_test)

            lst = [[0, 0, 0, 0.0, -1]]
            df_test = pd.DataFrame(lst, columns=["Type", "Equipment", "Level", "Rating", "RatingDesc"])
            # df_test

            y_sing = clf_gym.predict(df_test)
            # print("Result: ", y_sing[0], " ")

            # df.loc[df['column_name'] == some_value]

            # print(type(temp_df.loc[temp_df['New_ID'] == y_sing[0]]))

            # print("RES_NAME: ", temp_df.loc[temp_df['New_ID'] == y_sing[0]]['Title'])
            models.append(clf_gym)
            partial_dfs.append(temp_df)

        exercises = {}
        for i in range(len(models)):
            body_part = parts[i]
            model = models[i]
            table = partial_dfs[i]

            lst = [[goal, equip, level, 0.0, -1]]
            df_test = pd.DataFrame(lst, columns=["Type", "Equipment", "Level", "Rating", "RatingDesc"])
            # print(df_test)

            exercise = model.predict(df_test)
            id = table.index[table['New_ID'] == exercise[0]][0]
            # print("ID: ", id, ", RES_NAME: ", table.loc[table['New_ID'] == exercise[0]]['Title'] )
            # print("ID: ", id, ", RES_NAME: ", df_gym.iloc[[id]]['Title'].tolist()[0] )
            exercises[body_part] = df_gym.iloc[[id]]

        print("WRITE: ", len(exercises))

        self.mg_abdo.text = exercises[parts[0]]['Title'].tolist()[0]
        self.mg_addu.text = exercises[parts[1]]['Title'].tolist()[0]
        self.mg_abdu.text = exercises[parts[2]]['Title'].tolist()[0]
        self.mg_bice.text = exercises[parts[3]]['Title'].tolist()[0]
        self.mg_calv.text = exercises[parts[4]]['Title'].tolist()[0]
        self.mg_ches.text = exercises[parts[5]]['Title'].tolist()[0]
        self.mg_fore.text = exercises[parts[6]]['Title'].tolist()[0]
        self.mg_glut.text = exercises[parts[7]]['Title'].tolist()[0]
        self.mg_hams.text = exercises[parts[8]]['Title'].tolist()[0]
        self.mg_lats.text = exercises[parts[9]]['Title'].tolist()[0]
        self.mg_lbac.text = exercises[parts[10]]['Title'].tolist()[0]
        self.mg_mbac.text = exercises[parts[11]]['Title'].tolist()[0]
        self.mg_trap.text = exercises[parts[12]]['Title'].tolist()[0]
        self.mg_neck.text = exercises[parts[13]]['Title'].tolist()[0]
        self.mg_quad.text = exercises[parts[14]]['Title'].tolist()[0]
        self.mg_shou.text = exercises[parts[15]]['Title'].tolist()[0]
        self.mg_tric.text = exercises[parts[16]]['Title'].tolist()[0]

        hst = exercises
        # print(hst)

        columns = []
        for x in hst.values():
            y = x['Title'].tolist()[0]
            columns.append(f"{y}, reps")
            columns.append(f"{y}, weight (kg)")
            # columns.append(f"{y}, rest (s)")

        user_program = pd.DataFrame(columns=columns)
        user_program.insert(0, "Training_number", None)
        user_program.insert(0, "Date", None)

        # print(user_program)




class ContainerTest(Screen):
    pass

class Container2(GridLayout):

    text_input = ObjectProperty()
    label_widget = ObjectProperty()

    def change_text(self):
        self.label_widget.text = self.text_input.text.upper()

class MyApp(App):
    def build(self):


        sm = ScreenManager()
        sm.add_widget(InfoScreen(name='info'))
        sm.add_widget(ProgramScreen(name='program'))

        return sm


def calculate_BMI(weight, height):
    BMI = round(weight / (height/100)**2, 1)
    s = ""

    if BMI <= 18.4:
        s += "You are underweight."
    elif BMI <= 24.9:
        s += "You are healthy."
    elif BMI <= 29.9:
        s += "You are over weight."
    elif BMI <= 34.9:
        s += "You are severely over weight."
    elif BMI <= 39.9:
        s += "You are obese."
    else:
        s += "You are severely obese."

    return f"{BMI}, " + s

class ContainerFitApp(BoxLayout):
    def calculate_bmi_cal(self):
        try:
            age = int(self.text_input_age.text)
        except:
            age = 0

        print("TEST: ", self.text_input_height)
        try:
            height = int(self.text_input_height.text)
        except:
            height = 0

        try:
            weight = int(self.text_input_weight.text)
        except:
            weight = 0

        try:
            gender = int(self.text_input_gender.text)
            if gender != "M" or gender != "F":
                gender = "M"
        except:
            gender = "M"

        self.text_output_BMI.text = calculate_BMI(weight, height)

        self.text_output_calorie.text = "3000" if gender == "M" else "2400"



class FitApp(App):
    goals = ['Strength', 'Plyometrics', 'Cardio', 'Stretching', 'Powerlifting',
       'Strongman', 'Olympic Weightlifting']



    def build(self):

        sm = ScreenManager()
        sm.add_widget(InfoScreen(name='info'))
        sm.add_widget(ProgramScreen(name='program'))
        sm.add_widget(MapScreen(name='map'))
        sm.add_widget(HistoryScreen(name='history'))
        sm.add_widget(CalScreen(name='cal'))
        sm.add_widget(GraphsScreen(name='graphs'))
        sm.add_widget(GPSScreen(name='gps'))
        sm.add_widget(HeartRateScreen(name='hr'))

        return sm




if __name__ == "__main__":
    # MyApp().run()
    FitApp().run()


