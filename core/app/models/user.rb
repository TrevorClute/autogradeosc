class User < ApplicationRecord
  # We keep 'database_authenticatable' so Devise can handle sessions,
  # but we will generate random passwords behind the scenes.
  devise :database_authenticatable, :registerable, :rememberable, :validatable

  validates :email, format: {
    with: /\A[\w+\-.]+@osc\.edu\z/i,
    message: "must be an @osc.edu address"
  }

  # This method allows creating a user without a password (we generate a fake one)
  def password_required?
    false
  end

  # Helper to generate and save a 6-digit code
  def generate_otp!
    self.otp_code = rand(100000..999999).to_s
    self.otp_sent_at = Time.current
    save!
  end

  # Helper to check if code is valid (and not older than 10 mins)
  def valid_otp?(code)
    return false if otp_sent_at < 10.minutes.ago
    otp_code == code
  end
end
