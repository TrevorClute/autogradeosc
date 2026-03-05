class FlaskEvaluateService
  # Load secrets from .env
  FLASK_URL = ENV.fetch('FLASK_URL')
  API_KEY   = ENV.fetch('FLASK_API_KEY')

  def self.evaluate(text, prompt)
    conn = Faraday.new(url: FLASK_URL)

    response = conn.post('/evaluate') do |req|
      req.headers['Content-Type'] = 'application/json'
      req.headers['X-Internal-Secret'] = API_KEY
      req.body = { essay_text: text, prompt_id: prompt }.to_json
    end

    if response.success?
      JSON.parse(response.body)
    else
      { "error" => "Backend Error (Status: #{response.status})" }
    end
  rescue Faraday::ConnectionFailed
    { "error" => "Could not connect to OSC Backend. Is Python running?" }
  end

  def self.evaluate_batch(essays)
    conn = Faraday.new(url: FLASK_URL)

    response = conn.post('/evaluate/batch') do |req|
      req.headers['Content-Type'] = 'application/json'
      req.headers['X-Internal-Secret'] = API_KEY
      req.body = { essays: essays }.to_json
    end

    if response.success?
      JSON.parse(response.body)
    else
      { "error" => "Backend Error (Status: #{response.status})" }
    end
  rescue Faraday::ConnectionFailed
    { "error" => "Could not connect to OSC Backend. Is Python running?" }
  end
end
