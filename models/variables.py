feature_columns = {
    'text_static': [
        'script_scene_num',  'script_cell_num',  'script_action_num',  'script_dialog_num',  'script_action_num_per_cell_num',
        'script_dialog_num_per_cell_num',  'script_text_length',  'script_text_length_avg',  'script_text_length_std',
        'script_text_length_median',  'script_action_text_length',  'script_action_text_length_avg',  'script_action_text_length_std',
        'script_action_text_length_median',  'script_dialog_text_length',  'script_dialog_text_length_avg',  'script_dialog_text_length_std',
        'script_dialog_text_length_median', 'characters_num'
    ],
    'text_emotion': [
        'text_complaint',  'text_welcome',  'text_admiration',  'text_fed_up',  'text_gratitude',  'text_sadness',  'text_anger',
        'text_respect',  'text_anticipation',  'text_condescending',  'text_disappointment',  'text_resolute',  'text_distrust',
        'text_proud',  'text_comfort',  'text_fascination',  'text_caring',  'text_embarrassment',  'text_fear',  'text_despair',
        'text_pathetic',  'text_disgust',  'text_annoyance',  'text_dumbfounded',  'text_neutral',  'text_self_hatred',  'text_bothersome',
        'text_exhaustion',  'text_excitement',  'text_realization',  'text_guilt',  'text_hatred',  'text_fondness',  'text_flustered',
        'text_shock',  'text_reluctance',  'text_sorrow',  'text_boredom',  'text_compassion',  'text_surprise',  'text_happiness',
        'text_anxiety',  'text_joy',  'text_relief'
    ],
    'embedding': [f'feature_{i}' for i in range(3584)],
    'video_static': [
        'video_duration',  'video_cut_num',  'video_cut_min',  'video_cut_median',  'video_cut_max',  'video_cut_avg',  'video_cut_std'
    ],
    'video_emotion': [
        'video_complaint',  'video_welcome',  'video_admiration',  'video_fed_up',  'video_gratitude',  'video_sadness',  'video_anger',
        'video_respect',  'video_anticipation',  'video_condescending',  'video_disappointment',  'video_resolute',  'video_distrust',
        'video_proud',  'video_comfort',  'video_fascination',  'video_caring',  'video_embarrassment',  'video_fear',  'video_despair',
        'video_pathetic',  'video_disgust',  'video_annoyance',  'video_dumbfounded',  'video_neutral',  'video_self_hatred',  'video_bothersome',
        'video_exhaustion',  'video_excitement',  'video_realization',  'video_guilt',  'video_hatred',  'video_fondness',  'video_flustered',
        'video_shock',  'video_reluctance',  'video_sorrow',  'video_boredom',  'video_compassion',  'video_surprise',  'video_happiness',
        'video_anxiety',  'video_joy',  'video_relief'
    ],
    'audio_static': [
        'music_num',  'music_num_over10',  'music_duration',  'music_duration_avg',  'music_duration_std',  'music_duration_per_video_duration',
        'caption_row_num',  'caption_length',  'caption_avg'
    ],
    'music_mood': [
        'music_arousal',  'music_valence',  'music_action',  'music_adventure',  'music_advertising',  'music_background',  'music_ballad',
        'music_calm',  'music_children',  'music_christmas',  'music_commercial',  'music_cool',  'music_corporate',  'music_dark',  'music_deep',
        'music_documentary',  'music_drama',  'music_dramatic',  'music_dream',  'music_emotional',  'music_energetic',  'music_epic',
        'music_fast',  'music_film',  'music_fun',  'music_funny',  'music_game',  'music_groovy',  'music_happy',  'music_heavy',
        'music_holiday',  'music_hopeful',  'music_inspiring',  'music_love',  'music_meditative',  'music_melancholic',  'music_melodic',
        'music_motivational',  'music_movie',  'music_nature',  'music_party',  'music_positive',  'music_powerful',  'music_relaxing',
        'music_retro',  'music_romantic',  'music_sad',  'music_sexy',  'music_slow',  'music_soft',  'music_soundscape',  'music_space',
        'music_sport',  'music_summer',  'music_trailer',  'music_travel',  'music_upbeat',  'music_uplifting'
    ],
    'genere': [
        'genre_sf',  'genre_family',  'genre_horror',  'genre_other',  'genre_drama',  'genre_romance',  'genre_mystery',
        'genre_crime',  'genre_historical',  'genre_western',  'genre_thriller',  'genre_action',  'genre_adventure',  'genre_war',
        'genre_comedy',  'genre_fantasy'
    ],
    'rating': ['rating_all', 'rating_12', 'rating_15', 'rating_18'],
    'release_date': ['release_date_year',  'release_date_sin',  'release_date_cos',  'covid']
}