Rails.application.routes.draw do
  # 1. Setup Devise (Skip default routes)
  devise_for :users, skip: [:sessions, :registrations]

  # 2. Tell Devise where the "Sign In" page actually is
  devise_scope :user do
    get  '/login',  to: 'auth#login',      as: :new_user_session
    post '/login',  to: 'auth#send_code',  as: :send_code
    delete '/logout', to: 'auth#logout',   as: :destroy_user_session
  end

  # 3. Custom Verification Routes
  get  '/verify', to: 'auth#verify',      as: :verify
  post '/verify', to: 'auth#verify_code', as: :verify_code

  # 4. Main App Routes
  resources :essays, only: [:new, :create] do
    collection do
      get :batch
      post :batch, to: 'essays#create_batch'
    end
  end
  root "essays#new"

  match "*path", to: 'application#not_found', via: :all
end
