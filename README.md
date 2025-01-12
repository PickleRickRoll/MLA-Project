# Music Transcription Model Implementation

This project focuses on applying machine learning techniques to music transcription, specifically implementing the model described in the article *"A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation"* ([GitHub link](https://github.com/spotify/basic-pitch)). We did not create this model from scratch; instead, we replicated and implemented it based on the methods and architecture outlined in the paper.

Unlike many existing models that are often heavy, instrument-specific, or designed for monophonic inputs, the model we implemented demonstrates that a much simpler and lighter architecture can provide results comparable to or even better than some of the more complex models currently in use.

The main innovation of the referenced paper is the use of multiple outputs from a single model, with each output capturing different aspects of the music. This multi-output approach allows for more comprehensive transcription while keeping the model lightweight and adaptable to various instruments.



