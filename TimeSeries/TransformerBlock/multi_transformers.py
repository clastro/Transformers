def get_model(input_size=561, filt=3, dims=64, dff=32, kernel_size=3, num_heads=2, num_classes=6):
    input_tens = tf.keras.Input(shape=(input_size, 1))
    
    # Param branch
    output_param = SignalCNNEmbedding(filt=filt, sig_dims=dims)(input_tens)
    for i in range(8):
        output_param = TransformerBlockForAWOR(
            embedding_dim=dims,
            dff=dff,
            attention_layer=MultiHeadAttentionCycle_UseParams(
                embedding_dim=dims,
                kernel_size=kernel_size,
                sig_len = input_size,
                num_heads=num_heads,
                name=f"MultiHeadAttention_cycle_{i}"),
            name=f"Transformer_Cycle_Block_Enc_{i}")(output_param)
        
    output_param = tf.keras.layers.Flatten()(output_param)

    # Sig branch
    output_sig = input_tens
    for i in range(8):
        output_sig = TransformerBlockForAWOR(
            embedding_dim=dims,
            dff=dff,
            attention_layer=MultiHeadAttention(
                embedding_dim=dims,
                num_heads=num_heads,
                name=f"MultiHeadAttention_{i}"),
            name=f"Transformer_Block_Enc_{i}")(output_sig)
        
    output_sig = tf.keras.layers.Flatten()(output_sig)

    # Combine outputs
    combined_output = tf.keras.layers.Concatenate()([output_param, output_sig])
    
    # Optional: Add a dense layer to reduce dimensionality
    combined_output = tf.keras.layers.Dense(64, activation='relu')(combined_output)
    
    # Final classification layer
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(combined_output)

    model = tf.keras.Model(inputs=input_tens, outputs=output)
    print(model.summary())
    return model
