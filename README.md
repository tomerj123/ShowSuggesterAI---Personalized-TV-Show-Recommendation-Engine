ShowSuggesterAI - Personalized TV Show Recommendation Engine
a recommendation system that uses AI to provide personalized TV show suggestions based on user preferences.

For getting the recommended shows we will follow this logic:
○ The user has entered N shows that he liked and we matched every one
of his shows to a show in our popular show list.
○ We load the embedding vectors from the disk and we find the vector for
each show. Now we have N vectors (1 for each show).
○ We now calculate a new vector which is simply an average of the N
vectors.
○ We go over in a loop over all the 200 shows vectors and check the
distance of each vector to our average vector that we
calculated in the previous step. We find the 5 shows(which are not the
‘input shows’ of course) with the shortest distance to the average
vector. We sort them (shortest distance first) and we output to
the user the recommendations. 
○ then we create a new tv show- we use open ai chatgpt api using
prompts. 
For the creation of images we use openai’s dall-e generation api.
