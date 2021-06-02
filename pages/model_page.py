import time
import pandas as pd
import seaborn as sns
import streamlit as st 
from matplotlib.figure import Figure
from classes.process import FFT_Process, GNB_Process, SVM_Process



class model_page : 

    @staticmethod
    def bar_progress():
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.05)
            my_bar.progress(percent_complete + 1)

    @staticmethod
    def intro_gender():
        page_icon = "https://user-images.githubusercontent.com/46791116/119871124-0e9a9a00-bf1a-11eb-93c6-86b9011fd05c.png"
        description = f"""
            <div align='center'>
            <img src={page_icon}
            width="100" height="100">

            # Gender Recognition

            [![Twitter](https://badgen.net/badge/icon/Twitter?icon=twitter&label)](https://twitter.com/pytorch_ignite)
            [![GitHub](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/pytorch-ignite/code-generator)
            [![Release](https://badgen.net/github/tag/pytorch-ignite/code-generator/?label=release)](https://github.com/pytorch-ignite/code-generator/releases/latest)

            </div>

            ---
            """
        st.write(description, unsafe_allow_html=True)

    @staticmethod
    def intro_speaker():
        page_icon = "https://user-images.githubusercontent.com/46791116/119872106-0131df80-bf1b-11eb-8c4d-402cf5ea5d81.png"
        description = f"""
            <div align='center'>
            <img src={page_icon}
            width="100" height="100">

            # Speaker Recognition

            [![Twitter](https://badgen.net/badge/icon/Twitter?icon=twitter&label)](https://twitter.com/pytorch_ignite)
            [![GitHub](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/pytorch-ignite/code-generator)
            [![Release](https://badgen.net/github/tag/pytorch-ignite/code-generator/?label=release)](https://github.com/pytorch-ignite/code-generator/releases/latest)

            </div>

            ---
            """
        st.write(description, unsafe_allow_html=True)
        
    @staticmethod
    def intro_both():
        # https://user-images.githubusercontent.com/46791116/119872106-0131df80-bf1b-11eb-8c4d-402cf5ea5d81.png
        
        page_icon = "https://user-images.githubusercontent.com/46791116/119973531-81078a80-bfab-11eb-8c1f-a2efbcc710bb.png"
        description = f"""
            <div align='center'>
            <img src={page_icon}
            width="100" height="100">

            # Under Construction :/

            [![Twitter](https://badgen.net/badge/icon/Twitter?icon=twitter&label)](https://twitter.com/pytorch_ignite)
            [![GitHub](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/pytorch-ignite/code-generator)
            [![Release](https://badgen.net/github/tag/pytorch-ignite/code-generator/?label=release)](https://github.com/pytorch-ignite/code-generator/releases/latest)

            </div>

            ---
            """
        st.write(description, unsafe_allow_html=True)

    @staticmethod
    def voiceprints():
        '''
        This function returns an ay dist for the desired wr
        '''

        fig1 = Figure()
        ax = fig1.subplots()
        sns.kdeplot(data=[3,0,4,9,5,1], color='#CCCCCC',
                        fill=True, label='NFL Average',ax=ax)
        sns.kdeplot(data=[3,0,4,9,5,1],
                        fill=True, label='test',ax=ax)
        ax.legend()
        ax.set_xlabel('Air Yards', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.grid(zorder=0,alpha=.2)
        ax.set_axisbelow(True)
        ax.set_xlim([-10,55])
        st.pyplot(fig1)

    @staticmethod
    def show_speaker(info_speaker, perc_pred): 
        
        row1_1, row1_space2, row1_2, row1_space3 = st.beta_columns(
            (1, .05, 1, .00000001))
        with row1_1:
            url = info_speaker['Image'].values[0]
            print(' Debugging ...')
            print(url)
            st.subheader('Speaker Info')
            # st.write('### Speaker Info')
            st.subheader(' ')
            st.image(url, width=300)

        with row1_2:
            st.header(' ')
            st.text(' ')
            st.text(' ')
            st.text(
                f"Name: {info_speaker['Name'].values[0]} "
                )
            st.text(
             f"Percentage of Identification: {'%.2f' % (perc_pred*100)}%"
             )
            st.text(
                f"Nationality: "
            )
            st.text(f" ")
            # st.text(
            #     f"Birthday: "
            # )
            # desc_1 = f"""
            # <a href="https://www.w3schools.com/">More Info!</a> """
            # st.write(desc_1, unsafe_allow_html=True)
            model_page.voiceprints()

    @staticmethod 
    def DL_page(box): 
        if box == "FFT-Conv1D" : 
            perc_pred, id_speakers  = FFT_Process.get_prediction('output.wav')
            model_page.show_speaker(perc_pred, id_speakers)
        else : 
            st.write('# waiiiiit §')

    @staticmethod 
    def ML_page(box):
        
        actors_me = pd.read_csv("Speakers Info\\ravdess_support.csv")
        if box == 'MFCC-SVM' : 
            perc_pred, id_speakers  = SVM_Process.predict_svm()
            print(type(id_speakers[0]))
            print(int(id_speakers[0]))
            info_speaker= actors_me.loc[actors_me['Id']== id_speakers[0]]
            model_page.show_speaker(info_speaker, perc_pred)
        else : 
            perc_pred, id_speakers  = GNB_Process.predict_gnb()
            info_speaker= actors_me.loc[actors_me['Id'] == id_speakers[0]]
            model_page.show_speaker(info_speaker, perc_pred)

    @staticmethod
    def show(approach_selectbox, model_selectbox):
        model_page.intro_speaker()
        if approach_selectbox == 'Machine Learning':
            model_page.ML_page(model_selectbox)
        else : 
            model_page.DL_page(model_selectbox)  