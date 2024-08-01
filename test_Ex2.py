# Re-importing the necessary libraries and re-defining the test cases since the code execution state was reset.
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

import pytest
import ShowSuggesterAI  # Assuming this is the name of the file containing the provided code

mock_tv_shows_df = pd.DataFrame({
    'Title': ['Show1', 'Show2', 'Show3'],
    'Description': ['Desc1', 'Desc2', 'Desc3'],
    'Genres': ['Genre1', 'Genre2', 'Genre3']
})

# Mock embeddings dictionary
mock_embeddings = {
    'Show1': np.array([0.1, 0.2, 0.3]),
    'Show2': np.array([0.4, 0.5, 0.6]),
    'Show3': np.array([0.7, 0.8, 0.9])
}

@patch('pandas.read_csv', return_value=mock_tv_shows_df)
def test_load_tv_shows(mock_read_csv):
    result = ShowSuggesterAI.load_tv_shows('dummy/path.csv')
    assert not result.empty
    assert list(result.columns) == ['Title', 'Description', 'Genres']
    mock_read_csv.assert_called_with('dummy/path.csv')

# THIS IS TO LOAD TO PICKLE FIRST TIME
# def test_embeddings_loading():
#     embeddings = ShowSuggesterAI.load_embeddings('./embeddings.pkl','./imdb_tvshows - imdb_tvshows.csv' )
#     assert embeddings is not None
#     assert len(embeddings) > 0

def test_confirm_show_names():
    # Sample user input and TV shows DataFrame
    user_input = "The Witcher, Game of Thrones"
    shows_df = pd.DataFrame({
        'Title': ['The Witcher', 'Game of Thrones', 'Stranger Things']
    })
    result = ShowSuggesterAI.confirm_show_names(user_input, shows_df)
    assert result is not None
    assert isinstance(result, list)
    assert 'The Witcher' in result
    assert 'Game of Thrones' in result

def test_calculate_average_vector():
    # Mock embeddings and shows list
    embeddings = {'The Witcher': np.array([0.1, 0.2, 0.3]), 'Game of Thrones': np.array([0.4, 0.5, 0.6])}
    shows = ['The Witcher', 'Game of Thrones']
    result = ShowSuggesterAI.calculate_average_vector(shows, embeddings)
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)

def test_find_closest_shows():
    # Mock data for testing
    average_vector = np.array([0.25, 0.35, 0.45])
    embeddings = {'The Witcher': np.array([0.1, 0.2, 0.3]), 'Game of Thrones': np.array([0.4, 0.5, 0.6]),
                  'Stranger Things': np.array([0.2, 0.3, 0.4])}
    input_shows = ['The Witcher']
    result = ShowSuggesterAI.find_closest_shows(average_vector, embeddings, input_shows)
    assert result is not None
    assert isinstance(result, list)
    assert all(isinstance(item, tuple) for item in result)

def test_movie_creator():
    """
    Test the movie_creator function for generating a movie idea based on a list of show names.
    """
    confirm_shows = ["The Witcher", "Game of Thrones"]
    movie_idea = ShowSuggesterAI.movie_creator(confirm_shows)
    assert isinstance(movie_idea, str)
    assert "Name of the movie created:" in movie_idea

def test_image_creator():
    """
    Test the image_creator function to ensure it returns a valid URL for a movie ad image.
    """
    title = "Test Movie"
    description = "A test movie description."
    image_url = ShowSuggesterAI.image_creator(title, description)
    assert isinstance(image_url, str)
    assert image_url.startswith("http")

def test_movie_name_description_extract():
    """
    Test the movie_name_description_extract function to ensure it correctly extracts movie name and description from a generated idea.
    """
    movie_created = "Test Movie: A test movie description."
    extracted_info = ShowSuggesterAI.movie_name_description_extract(movie_created)
    assert isinstance(extracted_info, dict)
    assert "Test Movie" in extracted_info
    assert extracted_info["Test Movie"] == "A test movie description."

@patch('os.path.exists', return_value=False)
def test_load_embeddings_no_file(mock_exists):
    with patch('ShowSuggesterAI.generate_embeddings_for_shows', return_value=mock_embeddings):
        result = ShowSuggesterAI.load_embeddings('dummy.pkl', 'dummy.csv')
        assert result == mock_embeddings

# Test for error scenarios in generate_embeddings_for_shows
@patch('ShowSuggesterAI.client.embeddings.create', side_effect=Exception("API Error"))
def test_generate_embeddings_for_shows_error(mock_create):
    with pytest.raises(Exception) as excinfo:
        ShowSuggesterAI.generate_embeddings_for_shows('dummy.csv')
    assert "API Error" in str(excinfo.value)

# Test for different inputs in confirm_show_names
def test_confirm_show_names_no_match():
    user_input = "Unknown Show"
    result = ShowSuggesterAI.confirm_show_names(user_input, mock_tv_shows_df)
    assert not result == [user_input]

# Test for calculate_average_vector with missing shows in embeddings
def test_calculate_average_vector_missing_shows():
    movie_titles  = mock_tv_shows_df["Title"] 
    result = ShowSuggesterAI.calculate_average_vector(movie_titles, mock_embeddings)
    assert result is not None
    assert len(result) == 3

# Test for find_closest_shows with empty embeddings
def test_find_closest_shows_empty_embeddings():
    average_vector = np.array([0.25, 0.35, 0.45])
    result = ShowSuggesterAI.find_closest_shows(average_vector, {}, ['The Witcher'])
    assert result == []

# Mock tests for movie_creator and image_creator
@patch('ShowSuggesterAI.client.chat.completions.create', return_value=MagicMock())
def test_movie_creator_mocked_api(mock_api):
    result = ShowSuggesterAI.movie_creator(['The Witcher'])
    assert result is not None

@patch('ShowSuggesterAI.client.images.generate', return_value=MagicMock())
def test_image_creator_mocked_api(mock_api):
    result = ShowSuggesterAI.image_creator('Test Title', 'Test Description')
    assert result is not None


def test_confirm_show_names_fuzzy_match():
    user_input =  "sh1, ow2"  # Intentionally misspelled
    result = ShowSuggesterAI.confirm_show_names(user_input, mock_tv_shows_df)
    assert 'Show1' in result
    assert 'Show2' in result

# Test calculate_average_vector with an empty list
def test_calculate_average_vector_empty_list():
    result = ShowSuggesterAI.calculate_average_vector([], mock_embeddings)
    assert result is None


# Test for main function (integration test)
@patch('ShowSuggesterAI.get_favorite_shows', return_value="The Witcher, Game of Thrones")
@patch('ShowSuggesterAI.input', side_effect=['y'])  # Mocking user input for confirmations
@patch('ShowSuggesterAI.webbrowser.open')  # Mocking webbrowser open calls
def test_main(mock_webbrowser, mock_input, mock_get_favorite_shows):
    ShowSuggesterAI.main()
    mock_get_favorite_shows.assert_called_once()
    assert mock_input.call_count == 1
    assert mock_webbrowser.call_count == 2