import openai
import json

def generate_frame_descriptions(context, custom = True):
    if custom:
        openai.api_key = "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"
        start_chat_log = f"""
        The following is a detailed and vivid description for a video, split into six distinct frames:

        Context: Flower budding
        Output:
        ["A group of closed buds can be seen on the stem of a plant.",
        "The buds begin to slowly unfurl, revealing small petals.",
        "The petals continue to unfurl, revealing more of the flower's center.",
        "The petals are now fully opened, and the center of the flower is visible.",
        "The flower's stamen and pistil become more prominent, and the petals start to curve outward.",
        "The fully bloomed flowers are in full view, with their petals open wide and displaying their vibrant colors."]

        Context: Volcanic eruption
        Output:
        ["A towering volcano stands against a backdrop of clear blue skies, with no visible signs of activity.",
         "Suddenly, a plume of thick smoke and ash erupts from the volcano's summit, billowing high into the air.",
        "Molten lava begins to flow down the volcano's slopes, glowing brightly with intense heat and leaving a trail of destruction.",
        "Explosions rock the volcano as fiery projectiles shoot into the sky, scattering debris and ash in all directions.",
        "The eruption intensifies, with a massive column of smoke and ash ascending into the atmosphere, darkening the surrounding area.",
       " As the eruption reaches its peak, a pyroclastic flow cascades down the volcano's sides, engulfing everything in its path with hot gases, ash, and volcanic material."]

        "Please generate a frame-by-frame description using a consistent and detailed narrative style as provided above for the given context. 
        The purpose of these frames is to seamlessly create a coherent video storyline. Ensure minimal changes between frames to maintain narrative continuity.
        Ensure the number of frames stays between 6 and 12. The output must only be a Python list and nothing else"

        Context: {context}
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": start_chat_log}
            ],
            max_tokens=500
        )

        if isinstance(response.choices[0].message["content"], list):
            return response.choices[0].message["content"]
        else:
            return json.loads(response.choices[0].message["content"])
    else:
        import random
        import time
        
        lists = [
            ["The sky, partially cloudy, with faint hints of colors starting to emerge.",
             "A faint arch of colors becomes visible, stretching across the sky.",
             "The rainbow gains intensity as the colors become more vibrant and defined.",
             "The rainbow is now fully formed, displaying its classic arc shape.",
             "The colors of the rainbow shine brilliantly against the backdrop of the sky.",
             "The rainbow remains steady, its colors vivid and captivating as the rainbow decorates the sky."],
            ["A group of closed buds can be seen on the stem of a plant.",
             "The buds begin to slowly unfurl, revealing small petals.",
             "The petals continue to unfurl, revealing more of the flower's center.",
             "The petals are now fully opened, and the center of the flower is visible.",
             "The flower's stamen and pistil become more prominent, and the petals start to curve outward.",
             "The fully bloomed flowers are in full view, with their petals open wide and displaying their vibrant colors."],
            ["A towering volcano stands against a backdrop of clear blue skies, with no visible signs of activity.",
             "Suddenly, a plume of thick smoke and ash erupts from the volcano's summit, billowing high into the air.",
             "Molten lava begins to flow down the volcano's slopes, glowing brightly with intense heat and leaving a trail of destruction.",
             "Explosions rock the volcano as fiery projectiles shoot into the sky, scattering debris and ash in all directions.",
             "The eruption intensifies, with a massive column of smoke and ash ascending into the atmosphere, darkening the surrounding area.",
             "As the eruption reaches its peak, a pyroclastic flow cascades down the volcano's sides, engulfing everything in its path with hot gases, ash, and volcanic material."],
            ["A dense forest canopy is lush and vibrant, filled with the rich greens of late summer under a clear blue sky.",
             "The first hints of autumn appear as subtle yellow and orange tinges begin to dot the canopy, contrasting with the green.",
             "Leaves turn brighter shades of orange, red, and gold, creating a colorful mosaic that blankets the entire forest.",
             "A crisp autumn breeze causes leaves to gently fall, swirling through the air and slowly carpeting the forest floor.",
             "The forest is now ablaze with full autumn colors; red, orange, and yellow leaves dominate the landscape under a soft, overcast sky.",
             "As late autumn sets in, the trees are left bare with most leaves fallen, and a thin layer of frost begins to coat the now-visible forest floor."]
        ]
    random.seed(time.time())
    return random.choice(lists)

    
    