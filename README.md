
# DEEP IMPRESSIONS

Deep Impression is a facial recognition application that links together the social identities of an individual online. 

With a snap of a picture, you can aggregate the public online profiles of an individual through recongition of their one unchanging identity.

Instructions to Run:

- Have an iPhone and Macbook available.
- Load app into iPhone (using xCode).
- run server.js file using the command: 'node server.js'. This will run the server on your localhost
- Download ngrok (https://ngrok.com/)
- Run the command ngrok localhost
- An address will be produced; copy the address into the third line of the function UploadRequest (e.g let url = NSURL(string: "https://bff31b7a.ngrok.io/process"))
- Run the app.
