import os
import streamlit as st
from io import StringIO
from utils.preprocessing import preprocess
import altair as alt
from matplotlib import pyplot as plt
from numerize import numerize

from utils.helpers import (
    fetch_messages,
    fetch_stats, 
    fetch_active_users, 
    get_wordcloud, 
    get_timelines,
    timeline_stats,
    get_activity_map,
)

from utils.topic_model import get_topics

import seaborn as sns
sns.set()


PAGE_CONFIG = {"page_title":"App by Glad Nayak","page_icon":":smiley:","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)

def main():
    """
    Render UI on web app, fetch and display data using utils.py
    """
    st.title("WhatsApp Chat Analysis")
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if not uploaded_file:
            st.subheader('Upload exported text file to see analysis')
        
    if uploaded_file:
        try:
            # save the file, and pass the filename to preprocess function
            with open(os.path.join(".", uploaded_file.name),"wb") as f:
                f.write(uploaded_file.getbuffer())

            chats_with_date, chats = preprocess(uploaded_file.name)
        
        except:
            st.text('Failed to read the exported file. Try again')
            return -1

        # Get all users involved in conversation
        users = chats.id.unique().tolist()
        users.sort()
        users.insert(0, 'Overall')

        
        # 1. Show stats based on selected user
        user = st.sidebar.selectbox('Show analysis wrt', users)

        user_title = f'Showing Analysis for {user}' if user != 'Overall' else 'Showing Overall Analysis'
        st.subheader(user_title)

        headers = ['Members', 'Messages', 'Words', 'Media Uploaded', 'Links Shared']
        stats = [chats.id.nunique()]
        stats.extend(fetch_stats(chats, user))

        # don't show total members for personal conversations
        headers = headers[1:] if user != 'Overall' else headers
        stats = stats[1:] if user != 'Overall' else stats

        metrics = zip(headers, st.columns(len(headers)), stats)
        for header, column, stat in metrics:
            with column:
                st.metric(label=header, value=numerize.numerize(stat))


        # 2. display dataframe
        st.dataframe(fetch_messages(chats, user))


        # 3. plot activity map of user
        title = f'Activity map of {user}' if user != 'Overall' else 'Activity Map of Users'
        st.subheader(title)

        headers = ['Active Days', 'First Message on', 'Last Message on']
        for header, column, value in zip(headers, st.columns(3), timeline_stats(chats, user)):
            with column:
                st.metric(label=header, value=value)

        st.altair_chart(get_activity_map(chats, user), use_container_width=True)


        # 4. Show Overall stats
        if user == 'Overall':
            headers = ['Most Active', 'Most Active(%)']
            functions = [st.bar_chart, st.table]
            top_users, top_users_percent = fetch_active_users(chats)
        
            col1, col2 = st.columns(2)

            # plot most active users
            top_users = top_users[:10] 

            with col1:
                st.subheader(f'Top {len(top_users)} Active Users')
                bar_chart = alt.Chart(top_users).mark_bar(
                        cornerRadiusTopLeft=3,
                        cornerRadiusTopRight=3,
                ).encode(
                        x=alt.X('User:N', axis=alt.Axis(labelAngle=45)),
                        y='Messages:Q',
                        color=alt.Color('User:N', scale=alt.Scale(scheme='goldorange'), legend=None),
                )

                text = bar_chart.mark_text(
                            align='center',
                            dy=-5,
                            color='white'
                            ).encode(
                            text='Messages'
                        )
                            
                bar = (bar_chart + text).properties(height=400)
                st.altair_chart(bar, use_container_width=True)

            # plot percentage of active users
            with col2:
                st.subheader('Most Active Users(%)')
                st.dataframe(top_users_percent)


        # 5. Plot word clouds
        try:
            st.subheader('Word Cloud')
            wc = get_wordcloud(chats, user)
            fig, ax = plt.subplots()
            ax.imshow(wc)
            plt.axis('off')
            st.pyplot(fig)

        except:
            st.text("There's been lot of silence lately...")


        # 6. plot metrics on words
        # removed due to memory issue

        # 7. display timelines
        stats = get_timelines(chats_with_date, user)
        for timeline_plot in stats:
            st.plotly_chart(timeline_plot, use_container_width=True)

        
        # 8. display topics
        topics = get_topics(chats)
        if topics:
            st.subheader('Learning what members are talking about using Topic Modelling')
            for idx, wc in enumerate(topics):
                try:
                    st.subheader(f'Topic {idx+1}')
                    fig, ax = plt.subplots()
                    ax.imshow(wc)
                    plt.axis('off')
                    st.pyplot(fig)

                except:
                    st.text("")

if __name__ == '__main__':
	main()
