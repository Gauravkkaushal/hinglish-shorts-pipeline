# Privacy policy - Hinglish Shorts Pipeline

Last updated: 26 November 2026

Hinglish Shorts Pipeline ("the app") is open-source software that each person runs on their own computer. Nobody runs a
server for it, and the author receives no data from anybody who uses it.

**Google / YouTube.** The app asks for one permission, `youtube.upload`: to upload the videos it made to the YouTube channel
you choose, when you run the upload command or the daily task you set up. It does not read, change or delete anything else
in your Google account. The sign-in is stored only on your computer, in the file `youtube_token.json`. You can remove the
app's access at any time at [myaccount.google.com/permissions](https://myaccount.google.com/permissions) and by deleting that file.

**Instagram.** If you set it up, the app uses the access token you put into your own `.env` file to publish your videos to
your own account. The token stays on your computer.

**Other services.** To make a video, the app sends audio, transcripts and a few still pictures of it to Groq
([groq.com/privacy-policy](https://groq.com/privacy-policy)), with your own Groq key, and downloads public videos from
YouTube. What these services do with it is covered by their own policies.

**Contact.** Questions: open an issue at
[github.com/Gauravkkaushal/hinglish-shorts-pipeline/issues](https://github.com/Gauravkkaushal/hinglish-shorts-pipeline/issues).
