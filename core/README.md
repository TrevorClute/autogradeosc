web wrapper for python eval service

### Setup

```bash
bundle install
rails db:create db:migrate
rails s
```

### Environment variables

FLASK_URL=\<url to flask service\> <br>
FLASK_API_KEY=\<internal secret to eval service\> <br>
GMAIL_USERNAME=\<username for sending emails\> <br>
GMAIL_PASSWORD=\<passkey for email\> <br>
SEND_GRID_KEY=\<sendgrid api key\> <br>
