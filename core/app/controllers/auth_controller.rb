class AuthController < ApplicationController
  skip_before_action :authenticate_user!, only: [ :login, :send_code, :verify, :verify_code ]

  def login
  end

  def send_code
    email = params[:email].downcase.strip

    unless email.match?(/\A[\w+\-.]+@osc\.edu\z/i)
      flash[:alert] = "Access restricted to @osc.edu emails."
      return redirect_to new_user_session_path
    end

    @user = User.find_or_initialize_by(email: email)

    if @user.new_record?
      @user.password = Devise.friendly_token
      @user.save!
    end

    @user.generate_otp!
    UserMailer.send_otp(@user).deliver_now

    session[:auth_email] = email
    redirect_to verify_path
  end

  def verify
    if session[:auth_email].blank?
      redirect_to login_path
    end
  end

  def verify_code
    email = session[:auth_email]
    code = params[:otp_code]
    user = User.find_by(email: email)

    if user && user.valid_otp?(code)
      sign_in(user)
      session.delete(:auth_email)
      redirect_to root_path, notice: "Signed in successfully!"
    else
      flash[:alert] = "Invalid or expired code."
      render :verify
    end
  end

  def logout
    sign_out(current_user)
    redirect_to login_path, notice: "Logged out."
  end
end
