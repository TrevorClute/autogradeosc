web wrapper for python eval service

### Setup

```bash
bundle install
rails db:create db:migrate
rails s
```

### Environment variables

FLASK_URL=\<url to flask service\>
FLASK_API_KEY=<internal secret to eval service>
GMAIL_USERNAME=<username for sending emails>
GMAIL_PASSWORD=<passkey for email>
SEND_GRID_KEY=<sendgrid api key>
