class EssaysController < ApplicationController
  # before_action :authenticate_user! # Forces Login

  def new
    # Renders the input form
  end

  def create
    prompt = params[:prompt]
    
    begin
      clean_text = EssayParserService.parse(params[:essay_text], params[:essay_file])

      if clean_text.blank?
        flash[:alert] = "Please provide text or upload a PDF."
        return render :new, status: :unprocessable_entity
      end

      @result = FlaskEvaluateService.evaluate(clean_text, prompt)

      render :show

    rescue StandardError => e
      flash[:alert] = "Error: #{e.message}"
      render :new, status: :unprocessable_entity
    end
  end

  def batch
    # Renders the batch upload form
  end

  def create_batch
    files = params[:essay_files]
    prompts = params[:prompts] || []

    if files.blank? || files.reject(&:blank?).empty?
      flash[:alert] = "Please upload at least one file."
      return render :batch, status: :unprocessable_entity
    end

    begin
      essays = []

      files.each_with_index do |file, index|
        next if file.blank?

        clean_text = EssayParserService.parse(nil, file)
        next if clean_text.blank?

        filename = file.original_filename
        prompt_id = (prompts[index] || 1).to_i
        essays << { essay_text: clean_text, prompt_id: prompt_id, essay_id: filename }
      end

      if essays.empty?
        flash[:alert] = "No valid essays found in uploaded files."
        return render :batch, status: :unprocessable_entity
      end

      response = FlaskEvaluateService.evaluate_batch(essays)

      if response["error"]
        flash[:alert] = response["error"]
        return render :batch, status: :unprocessable_entity
      end

      @results = response["results"]
      render :batch_results

    rescue StandardError => e
      flash[:alert] = "Error: #{e.message}"
      render :batch, status: :unprocessable_entity
    end
  end
end
