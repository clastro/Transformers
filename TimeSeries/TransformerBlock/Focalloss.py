class FocalLoss(tf.keras.losses.Loss):
    
    def __init__(self, gamma=2.0, alpha=0.25, num_classes=6, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), self.num_classes)
        cross_entropy_loss = -y_true_one_hot * tf.math.log(y_pred)
        focal_loss = self.alpha * tf.pow(1 - y_pred, self.gamma) * cross_entropy_loss
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))
