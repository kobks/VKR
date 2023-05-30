from kivy.app import App
from kivy.lang import Builder
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.scrollview import ScrollView

from kivy.core.window import Window

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics

# Window.size = (720, 1280)
Window.size = (480, 720)

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


class ProgramScreen(Screen):
    def generate_program(self):
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
        # for i in exercises.keys():
        #   print(f"For bodypart {i}, recommend { exercises.get(i)['Title'].tolist()[0]}")

        # self.program = exercises
        #
        # # columns = [x['Title'].tolist()[0] for x in self.program.values()]
        # columns = []
        # for x in self.program.values():
        #     y = x['Title'].tolist()[0]
        #     columns.append(f"{y}, reps")
        #     columns.append(f"{y}, weight (kg)")
        #     columns.append(f"{y}, rest (s)")
        #
        # # print("TEST: ", columns)
        #
        # self.history = pd.DataFrame(columns=columns)
        # self.history.insert(0, "Training_number", None)
        # self.history.insert(0, "Date", None)


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

        return sm




if __name__ == "__main__":
    # MyApp().run()
    FitApp().run()


