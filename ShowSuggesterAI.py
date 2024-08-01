import webbrowser
import pandas as pd
import numpy as np
import pickle
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import logging
import json


logging.basicConfig(level=logging.INFO, format='%(message)s')



client = OpenAI(
# this is a public project. The API key is sensitive information.
    organization="org-key",
    api_key="openaikey",
)


# For debugging purposes
# csv_file_path = 'assigment2_excercise2/imdb_tvshows - imdb_tvshows.csv'
# embeddings_file_path = 'assigment2_excercise2/embeddings.pkl'


csv_file_path = './imdb_tvshows - imdb_tvshows.csv'
embeddings_file_path = './embeddings.pkl'


# Function to create an Annoy index, importfor usearch and annoy didnt work.
# def create_annoy_index(embeddings, dimensions):
#     index = AnnoyIndex(dimensions, 'angular')  # Using angular distance
#     for i, (title, vector) in enumerate(embeddings.items()):
#         index.add_item(i, vector)
#     index.build(10)  # 10 trees for the index
#     return index

# def find_closest_shows(annoy_index, query_vector, embeddings, input_shows, top_k=3):

    # Find closest shows using the Annoy index, excluding shows in input_shows.
    # :param annoy_index: Annoy index containing the show embeddings.
    # :param query_vector: The average vector to compare against.
    # :param embeddings: Dictionary of show titles and their embeddings.
    # :param input_shows: List of shows to exclude from the results.
    # :param top_k: Number of closest shows to return.
    # :return: List of tuples (show title, similarity score) for the closest shows.

    # all_nearest_ids = annoy_index.get_nns_by_vector(query_vector, len(embeddings), include_distances=True)
    # closest_shows = []
    # for i, distance in zip(*all_nearest_ids):
    #     show_title = list(embeddings.keys())[i]
    #     if show_title not in input_shows:
    #         closest_shows.append((show_title, 1 - distance))
    #         if len(closest_shows) == top_k:
    #             break
    # return closest_shows


def load_tv_shows(path):
    # Load TV shows data
    tv_shows = pd.read_csv(path)
    return tv_shows

def load_embeddings(embeddings_file, csv_file):
    # Load embeddings if already exists, else create an embedding file
    if os.path.exists(embeddings_file) and os.path.getsize(embeddings_file) > 0:
        with open(embeddings_file, 'rb') as file:
            embeddings = pickle.load(file)
    else:
        embeddings = generate_embeddings_for_shows(csv_file)
        # Save embeddings to a file using pickle
        with open(embeddings_file, 'wb') as file:
            pickle.dump(embeddings, file)

    return embeddings

def generate_embeddings_for_shows(csv_file):
    # Load the CSV file containing TV shows
    tv_shows = pd.read_csv(csv_file)
    # Dictionary to store embeddings
    embeddings = {}
    # Generate embeddings for each show
    for _, row in tv_shows.iterrows():
        text = row["Description"] + " " + row["Genres"] 
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embeddings[row["Title"]] = response.data[0].embedding

    return embeddings


# Function to get favorite shows from user input
# def get_favorite_shows():
    return input("Which TV shows did you love watching? Separate them by a comma.\nMake sure to enter more than 1 show:\n")

def get_favorite_shows():
    logging.info("Which TV shows did you love watching? Separate them by a comma.\nMake sure to enter more than 1 show:")
    return input()


# Function to confirm show names using fuzzy string matching
def confirm_show_names(user_shows, tv_shows):
    user_shows_list = [show.strip() for show in user_shows.split(',')]
    confirmed_shows = []
    for show in user_shows_list:
        match = process.extractOne(show, tv_shows['Title'], score_cutoff=63)
        if match:
            confirmed_shows.append(match[0])
        else:
            # Get a list of potential matches
            potential_matches = process.extract(show, tv_shows['Title'], limit=3, scorer=process.fuzz.token_sort_ratio)
            # Select the best match from potential matches
            best_match = max(potential_matches, key=lambda x: x[1])[0] if potential_matches else show
            confirmed_shows.append(best_match)
    logging.info(f"Just to make sure, do you mean {', '.join(confirmed_shows)}?")
    confirmation = input("Please confirm (y/n):\n")
    return confirmed_shows if confirmation.lower() == 'y' else None

# Function to calculate the average embedding vector
def calculate_average_vector(shows, embeddings):
    vectors = [embeddings[show] for show in shows if show in embeddings]
    return np.mean(vectors, axis=0) if vectors else None

# Function to find the closest shows based on the average vector
def find_closest_shows(average_vector, embeddings, input_shows):
    distances = {}
    for show, vector in embeddings.items():
        if show not in input_shows:
            similarity = cosine_similarity([average_vector], [vector])[0][0]
            distances[show] = similarity
    sorted_shows = sorted(distances.items(), key=lambda x: x[1], reverse=True)
    return sorted_shows[:3]

# Function to display the recommendations
def display_recommendations(recommended_shows):
    logging.info("Here are the TV shows that I think you would love:\n")
    for show, score in recommended_shows:
        logging.info(f"{show} ({score*100:.0f}%)\n")

def movie_creator(confirm_shows):    
    system_prompt_movie_generator = ("Based on the following list of movies you will receive, generate new movie ideas."
     " For each movie idea, provide a unique name and a brief description."
    "Ensure the ideas are inspired by the themes, genres, and styles of the provided movies. Format the output as follows:"
    "name of the movie created: [Brief Description of the movie created]"
    )
    movies_list_str = ' and '.join(confirm_shows)

    user_movie_prompt =(f'Create for me a single movie similar to the following list of movies: {movies_list_str}, output 1 movie only, life and death, do not fail me, output should be in the following format: name of the movie created: [Brief Description of the movie created]')
    movies_gpt =client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_movie_prompt}, {"role": "system","content": system_prompt_movie_generator} ] ,
            temperature= 0.85,
            )
    movie_generator = movies_gpt.choices[0].message.content
    return movie_generator

 

def image_creator(title, description):
 

    user_prompt = (f"Engaging promotional art for a TV show ad for this movie: {title} based on this detailed description.: {description}"
                    "Highlight [central themes or settings], capturing the movie's atmosphere. Created Using: artistic style suited to the movie's tone,"
                   "vivid imagery, mood-driven color palette, and attention to narrative detail --ar 16:9 --v 6.0")
                                                     

    show_ad = client.images.generate(
    model="dall-e-3",
    prompt=user_prompt,
    size="1024x1024",
    quality="standard",
    n=1,
    )
    image_url = show_ad.data[0].url
   
    return image_url

def movie_name_description_extract(movie_created):
  
# Part 1: Task Explanation
    task_explanation = (
        "Upon receiving a movie idea, which includes a movie title and its description, "
        "analyze and organize the information into an object. "
    )

    # Part 2: Object Format Explanation
    object_format = (
        "The format of the object should be as follows in a json file: "
        "{ 'Movie Title': 'Movie Description' }. "
        "This object should use the movie title as the key and the movie description as the corresponding value. "
        "If additional movie ideas are provided, each should be added as a separate key-value pair within the object."
    )

    # Part 3: Combine and Integrate the Movie Ideas
    user_movie_prompt = task_explanation + object_format + f"movie ideas: {movie_created}, return a JSON object"

  

    movies_gpt =client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={ "type": 'json_object' },
            messages=[{"role": "user", "content": user_movie_prompt}, {"role": "system", "content": "You are a helpful assistant designed to output JSON."},] ,
            temperature= 0.1,
            )
    movie_extracted = movies_gpt.choices[0].message.content
    return  json.loads(movie_extracted) 


  


# Main function to run the ShowSuggesterAI program
def main():
    tv_shows = load_tv_shows(csv_file_path)
    embeddings = load_embeddings(embeddings_file_path, csv_file_path)
    # index = create_annoy_index(embeddings, dimensions=len(next(iter(embeddings.values()))))
    while True:
        user_shows = get_favorite_shows()
        confirmed_shows = confirm_show_names(user_shows, tv_shows)
        if confirmed_shows:
            logging.info("Great! Generating recommendations...\n")
            break
        else:
            logging.info("Sorry about that. Let's try again, please make sure to write the names of the tv shows correctly.\n")

    first_movie = movie_creator(confirmed_shows)
    first_extracted_movie_json = movie_name_description_extract(first_movie)
    first_extracted_movie = list(first_extracted_movie_json.items())[0]
    first_ad = image_creator(first_extracted_movie[0], first_extracted_movie[1])
    average_vector = calculate_average_vector(confirmed_shows, embeddings)
    recommended_shows = find_closest_shows( average_vector, embeddings, confirmed_shows)
    movie_recommended_names = [show[0] for show in recommended_shows]
    second_movie = movie_creator(movie_recommended_names)
    second_extracted_movie_json = movie_name_description_extract(second_movie)
    second_extracted_movie = list(second_extracted_movie_json.items())[0]
    second_ad = image_creator(second_extracted_movie[0], second_extracted_movie[1])
    display_recommendations(recommended_shows)

    logging.info(f"I have also created just for you two shows which I think you would love.\n"
                 f"Show #1 is based on the fact that you loved the input shows that you gave me. "
                 f"Its name is {first_extracted_movie[0]} and it is about {first_extracted_movie[1]}.\n"
                 f"Show #2 is based on the shows that I recommended for you. Its name is {second_extracted_movie[0]} "
                 f"and it is about {second_extracted_movie[1]}.\n"
                 f"Here are also the 2 tv show ads. Hope you like them!"
                 )
    if first_ad:  # Ensure the URL is not empty
        webbrowser.open(first_ad)
    if second_ad:  # Ensure the URL is not empty
        webbrowser.open(second_ad)





# main()

