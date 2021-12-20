import streamlit as st
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import spacy
import emoji
import altair as alt
from urlextract import URLExtract
import plotly.graph_objects as go

urlextractor = URLExtract()
spacy.cli.download("en")
nlp = spacy.load('en_core_web_sm')


def fetch_messages(df, user):
    """
    Returns messages of selected user
    """
    if user.lower() != 'overall':
        df = df[df.id == user]

    df = df[df.message != '<Media omitted>']
    df = df[df.message != 'This message was deleted']

    return df




def fetch_stats(df, user):
    """
    Returns stats on number of messages, members, media files, links shared
    """
    if user.lower() != 'overall':
        df = df[df.id == user]

    # 1. fetch number of messages
    num_messages = df.message.shape[0]

    # 2. count number of words, and
    # 3. number of urls
    words = []
    urls = []
    for message in df.message:
        words.extend(message)
        urls.extend(urlextractor.find_urls(message))

    num_words = len(words)
    num_links = len(urls)

    # 4. number of media files shared
    num_medias = df[df.message == '<Media omitted>'].shape[0]
    
    return num_messages, num_words, num_medias, num_links




def fetch_active_users(df):
    """
    Return dataframe on most active users
    """
    new_df = (df['id'].value_counts()
                    .reset_index()
                    .rename(columns={'index': 'User', 'id':'Messages'})
                    .sort_values(by='Messages', ascending=False)
    )

    active_users_percent = ( (df.id.value_counts()/int(df.shape[0]) * 100)
                                .apply(lambda x: f'{x:.2f}')
                                .reset_index()
                                .rename(columns={'index': 'User', 'id':'Messages(%)'})
                                .sort_values(by='Messages(%)', ascending=False)
                            )

    return new_df, active_users_percent




def get_wordcloud(df, user):
    """
    Generates word cloud
    """
    if user.lower() != 'overall':
        df = df[df.id == user]

    df = df[df.message != '<Media omitted>']
    df = df[df.message != 'This message was deleted']

    wc = WordCloud(width=700, height=300, min_font_size=12, background_color='white')
    wc = wc.generate(df['message'].str.cat(sep=' '))
    return wc





def most_common_words(df, user, n=10):
    """
    Tokenize each message, and build list of nouns, verbs, phrases 
    to return "n" most common words, and emojis
    """
    if user.lower() != 'overall':
        df = df[df.id == user]

    df = df[df.message != '<Media omitted>']
    df = df[df.message != 'This message was deleted']
    
    # tokenize
    tokens = [word for msg in df.message for word in msg.split()]
    doc = nlp(' '.join(tokens))

    # filter all tokens that arent stop words or punctuations
    words = [token.text
            for token in doc
            if not token.is_stop and not token.is_punct]

    # filter noun tokens that aren't stop words or punctuations
    nouns = [token.text
            for token in doc
            if (not token.is_stop and
                not token.is_punct and
                token.pos_ == "NOUN")]

    # filter verb tokens that aren't stop words or punctuations
    verbs = [token.text
            for token in doc
            if (not token.is_stop and
                not token.is_punct and
                token.pos_ == "VERB")]

    # filter emojis
    emojis = [word for word in words if word in emoji.UNICODE_EMOJI['en']]

    common_nouns = Counter(nouns).most_common(n)
    common_phrases = Counter(doc.noun_chunks).most_common(n)
    common_verbs = Counter(verbs).most_common(n)
    common_emojis = Counter(emojis).most_common(n)

    # create a dataframe and build a barchart
    def to_barchart(table):
        df = pd.DataFrame(table)
        df = df.rename(columns={0: 'Phrases', 1:'Count'})
        return _get_barchart(df, 'Count','Phrases', 'Phrases', 'Count')

    return to_barchart(common_nouns), to_barchart(common_verbs), to_barchart(common_emojis)




def _get_barchart(df, x, y, color, label):
    """
    helper function to build a barchart
    """
    bar_chart = alt.Chart(df).mark_bar(
                        cornerRadiusTopLeft=3,
                        cornerRadiusTopRight=3,
                ).encode(
                        x=alt.X(x, axis=alt.Axis(title=None)),
                        y=alt.Y(y, axis=alt.Axis(title=None)),
                        color=alt.Color(color, legend=None),
                )

    text = bar_chart.mark_text(
                align='center',
                dx=9,
                color='white'
                ).encode(
                text=label
            )
    return bar_chart + text




def timeline_stats(df, user):
    """
    Return timespan of messages, first message date, and last message date
    """
    if user.lower() != 'overall':
        df = df[df.id == user]

    df = df[df.message != '<Media omitted>']
    df = df[df.message != 'This message was deleted']

    if user.lower() == 'overall':
        total_days = df.Elapsed.max() 
    else:
        total_days = df.groupby(df.Elapsed)['message'].count().shape[0]

    first_date = df.iloc[0, [7, 5, 4]].to_dict()
    first_date = f"{first_date['Day']}-{first_date['Month']}-{first_date['Year']}"

    last_date = df.iloc[-1, [7, 5, 4]].to_dict()
    last_date = f"{last_date['Day']}-{last_date['Month']}-{last_date['Year']}"

    return total_days, first_date, last_date




def get_timelines(df, user):
    """
    Build line chart to showcase yearly timeline, 
    and chart charts to show  most active months, day of week, and hour of day
    """
    if user.lower() != 'overall':
        df = df[df.id == user]

    df = df[df.message != '<Media omitted>']
    df = df[df.message != 'This message was deleted']

    # timelines
    yearly_timeline = df.resample('M')['message'].count().reset_index()
    yearly_text = df.resample('M')['message'].apply(pd.Series.mode).tolist()

    df_emojis = df['message'].apply(lambda lst:[x if x in emoji.UNICODE_EMOJI['en'] else -1 for x in lst][0])
    df_emojis = df[df_emojis!=-1]

    levels = ['datetime']
    for idx in range(df_emojis.index.nlevels-1):
        levels.append(idx)

    df_emojis = df_emojis['message'].resample('M').apply(pd.Series.mode).reset_index(level=levels)

    daily_timeline = df.resample('D')['message'].count().reset_index()
    daily_timeline = daily_timeline[daily_timeline['message'] != 0]
    daily_text = df.resample('D')['message'].apply(pd.Series.mode).tolist()

    # most active days, and hours
    hourly_timeline = df.groupby([df.index.hour])['message'].count().reset_index()
    hourly_text = df.groupby([df.index.hour])['message'].apply(pd.Series.mode).reset_index()['message'].tolist()

    weekly_timeline = df.groupby([df.index.day_name()])['message'].count().reset_index()
    weekly_text = df.groupby([df.index.day_name()])['message'].apply(pd.Series.mode).reset_index()['message'].tolist()

    # yearly timeline displaying total messages in each month-year
    monthly_fig = go.Figure()
    monthly_fig.add_trace(go.Scatter(
        x=yearly_timeline['datetime'], 
        y=yearly_timeline['message'],
        hovertemplate =
        '<b>%{y:.2s} messages</b>: '+
        '<br><i>%{text}</i>',
        text = yearly_text,
        name='Timeline of emoticons'
    ))

    monthly_fig.add_trace(go.Bar(
        x=yearly_timeline['datetime'], 
        y=yearly_timeline['message'],
        hovertemplate="%{y:.2s}",
        name='Number of Messages',
        marker=dict(color=yearly_timeline['message'], colorbar=None),
    ))

    emoji_title = 'Timeline of messages (month-wise)' 
    if len(df_emojis) >= len(yearly_timeline):
        emoji_title = 'Evolution of Emoticons over time'
        

    monthly_fig.update_layout(
                        title=emoji_title,
                        yaxis= go.layout.YAxis(title="Total Messages"),
                        showlegend=False,
                        xaxis = go.layout.XAxis(title='Months', tickangle=45),
                        xaxis_tickformat = '%B<br>%Y',
                        autosize=False,
                        height=500,
                   )
    
    monthly_fig.update_xaxes(
        rangeslider_visible=True,
        )
    

    monthly_fig.add_trace(go.Scatter(
        x=df_emojis['datetime'], 
        y=yearly_timeline.set_index('datetime').loc[df_emojis.datetime, 'message'],
        text=df_emojis['message'].tolist(),
        mode="markers+text",
        name='',
    ))

    # daily timeline of messages, displays hover text
    daily_fig = go.Figure([
                           go.Scatter(
                               x=daily_timeline['datetime'], 
                               y=daily_timeline['message'],
                               hovertemplate = 
                               '<b>%{y:.2s} messages</b>:' +
                               '<br><i>%{text}</i>',
                                text = daily_text,
                                name='',
    )])

    daily_fig.update_layout(
                        title='Timeline of messages (day-wise)',
                        yaxis= go.layout.YAxis(title="Total Messages"),
                        showlegend=False,
                        xaxis = go.layout.XAxis(title='Days', tickangle=45),
                        xaxis_tickformat = '%d %B (%a)<br>%Y',
                        autosize=False,
                        height=500,
                   )
    
    daily_fig.update_xaxes(
        rangeslider_visible=True,
        )
    
    
    # most active days
    weekly_fig = go.Figure([go.Bar(
                                x=weekly_timeline['datetime'], 
                                y=weekly_timeline['message'],
                                hovertemplate =
                                   '<b>%{y:.2s} messages</b>: '+
                                   '<br><i>%{text}</i>',
                                text = weekly_text,
                                name='',
                                marker=dict(color=weekly_timeline['message'], colorbar=None),
                                
                )])
    weekly_fig.update_layout(
                        title='Most Active Days',
                        yaxis= go.layout.YAxis(title="Total Messages"),
                        showlegend=False,
                        xaxis = go.layout.XAxis(title='Day of the Week', tickangle=45)
                   )
    
    weekly_fig.update_traces(texttemplate='%{y:.2s}', textposition='outside')

    
    # most active hours
    hourly_fig = go.Figure([go.Bar(
                                x=hourly_timeline['datetime'], 
                                y=hourly_timeline['message'],
                                hovertemplate = 
                                    '<b>%{y:.2s} messages</b>: '+
                                    '<br><i>%{text}</i>',
                                text = hourly_text,
                                name='',
                                marker=dict(color=hourly_timeline['message'], colorbar=None),
                                
                )])
    hourly_fig.update_layout(
                        title='Most Active Hours',
                        yaxis= go.layout.YAxis(title="Total Messages"),
                        showlegend=False,
                        xaxis = go.layout.XAxis(title='Hours', tickangle=45, tickvals=list(range(24)))
                   )
    
    hourly_fig.update_traces(texttemplate='%{y:.2s}', textposition='outside')

    return  monthly_fig, daily_fig, weekly_fig, hourly_fig



def get_activity_map(df, user):
    """
    Plot activity map for each day and hour
    """
    if user.lower() != 'overall':
        df = df[df.id == user]

    df['period'] = df['hour'].astype(str) + '-' + ((df['hour'] + 1) % 24).astype(str)
    df = df.groupby(['DayName', 'period'])['message'].count().reset_index()
    df = df.fillna(1)
    # pivot = df.pivot_table(index='DayName', columns='period', values='message', aggfunc='count').fillna(0)
    fig = alt.Chart(df).mark_rect().encode(
            alt.X('period:O', axis=alt.Axis(title='hours')),
            alt.Y('DayName:O', axis=alt.Axis(title='days')),
            alt.Color('message:Q', scale=alt.Scale(scheme='goldorange'))
        )
    return fig
