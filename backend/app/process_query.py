"""
Updated process_query function that uses the remote model server.
This file should be used as a reference to update the main.py file.
"""

@app.post("/api/query", response_model=Dict[str, Any])
@rate_limit_ip_and_tokens(settings.api.rate_limit_calls)
async def process_query(request: QueryRequest, request_obj: Request):
    """Process a query using the appropriate model server"""
    global model_status
    
    # Get client info for tracking and rate limiting
    client_info = get_client_info(request_obj)
    
    # Validate and sanitize the query
    sanitized_query = sanitize_input(request.query)
    is_valid, error_message = validate_query(sanitized_query)
    
    if not is_valid:
        logger.warning(f"Invalid query from client {client_info['client_id']}: {error_message}")
        raise HTTPException(status_code=400, detail=error_message)
    
    # Track processing for performance metrics
    start_time = time.time()
    search_results = []
    
    try:
        # Determine if this is a code-related query
        query_category = categorize_query(sanitized_query)
        is_code_question = query_category.get("is_code_question", False)
        
        # Get context - either from request or web search
        context = request.context
        
        # If web search is enabled and no context is provided, fetch context from the web
        if request.web_search and settings.web_search.enabled and (not context or context.strip() == ""):
            search_query = request.search_query or sanitized_query
            logger.info(f"Performing web search for: {search_query}")
            context = await search_web(search_query)
            search_results = SEARCH_CACHE.get(search_query.lower().strip(), {}).get("results", [])
        
        if not context or context.strip() == "":
            context = settings.ai_response.default_context
        
        # Log the input
        logger.info(f"Model input: query_length={len(sanitized_query)}, context_length={len(context)}")
        
        # Prepare model input
        model_input = {
            'query_text': sanitized_query,
            'passage_text': context
        }
        
        # Set a timeout for the prediction
        try:
            # Check which model server to use
            use_remote = getattr(app.state, 'use_remote_server', False) and REMOTE_MODEL_SERVER_AVAILABLE
            use_jupyter = getattr(app.state, 'use_jupyter_server', False) and JUPYTER_MODEL_SERVER_AVAILABLE
            
            with asyncio.timeout(settings.model.predict_timeout_seconds):
                if use_remote:
                    # Use Remote model server for prediction
                    logger.info("Using Remote model server for prediction")
                    prediction_result = await predict_with_remote_server_async(model_input)
                elif use_jupyter:
                    # Use Jupyter model server for prediction
                    logger.info("Using Jupyter model server for prediction")
                    prediction_result = await predict_with_jupyter_server(model_input)
                else:
                    # No model server available
                    logger.error("No model server available - cannot process query")
                    raise HTTPException(status_code=503, detail="AI model server not available. Please try again later.")
                
                # Check for errors
                if "error" in prediction_result:
                    logger.error(f"Model server prediction error: {prediction_result['error']}")
                    raise Exception(f"Model server prediction error: {prediction_result['error']}")
                
                # Convert to expected format
                prediction = {
                    'start_index': prediction_result.get('start_index', 0),
                    'end_index': prediction_result.get('end_index', 0),
                    'confidence': prediction_result.get('confidence', 0.0)
                }
        
        except asyncio.TimeoutError:
            logger.error(f"Model prediction timed out after {settings.model.predict_timeout_seconds} seconds")
            raise HTTPException(
                status_code=503, 
                detail=f"AI model is taking too long to respond. Please try a simpler query."
            )
        
        # Extract the answer from the prediction
        result = extract_answer(
            prediction, 
            context, 
            query=sanitized_query,
            include_confidence=True
        )
        
        # Determine intent from the query
        intent = determine_intent(sanitized_query)
        
        # Build the response
        response = {
            "answer": result["answer"],
            "intent": intent,
            "confidence": result["confidence"],
            "is_code_question": is_code_question,
            "context_used": True if (context and context != settings.ai_response.default_context) else False,
            "method": result["method_used"],
            "processing_time": time.time() - start_time
        }
        
        # Log the interaction for monitoring
        log_query_response(
            sanitized_query, 
            result["answer"], 
            metadata={
                "client_id": client_info["client_id"],
                "intent": intent,
                "confidence": result["confidence"],
                "method_used": result["method_used"],
                "processing_time": time.time() - start_time,
                "is_code_question": is_code_question,
                "context_used": bool(context and context != settings.ai_response.default_context)
            }
        )
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Log the failed interaction
        log_query_response(
            sanitized_query, 
            "ERROR", 
            metadata={
                "client_id": client_info["client_id"],
                "error": str(e),
                "processing_time": time.time() - start_time
            }
        )
        
        raise HTTPException(status_code=500, detail=f"Error processing your request: {str(e)}")

