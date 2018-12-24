from django.test import TestCase
import pickle
import pandas as pd
import random

from model_mommy import mommy


class TestQuestion(TestCase):
    def setUp(self):
        
        self.models = mommy.make('polls.Question')

    def test_str(self):
        #https://docs.djangoproject.com/en/2.0/topics/testing/tools/#assertions
        self.assertEquals(str(self.models), self.models.question_text)

    def test_question_and_date(self):
        self.assertTrue(self.models.question_text in self.models.question_and_date())

class TestQuestionML(TestCase):
    def setUp(self):
        df = pd.read_csv('Historical_Data.csv')
        df = df['Questions']
        self.models = df.ix[random.sample(one_hot.index, n)]
        
    def test_ml_cases(self):
        loaded_model = pickle.load(open('logreg', 'rb'))
        col = pickle.load(open('encoderQuestions', 'rb'))
        
        # Create a test dataframe
        test_questions['Questions'] = str(self.models)
        test_questions['date_diff'] = 0
        test_questions['len'] = len(str(self.models))
        test_questions = col.transform(test_questions)
        pred_proba = loaded_model.predict_proba(test_questions)[::,1]
        threshold = 0.50
        if pred_proba > 0.50:
            self.assertEquals(str(self.models), self.models.question_text)
        else:
            pass

class TestChoice(TestCase):
    def setUp(self):
        self.models = mommy.make('polls.Choice')

    def test_str(self):
        self.assertEquals(str(self.models), self.models.choice_text)

    def test_add_vote(self):
        current_votes = self.models.votes
        self.models.add_vote()
        new_votes = current_votes + 1
        self.assertEquals(self.models.votes, new_votes)
