/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static default_random_engine random_gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	//set up the number of particles
	num_particles = 100;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  //init all particles to first position
  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(random_gen);
    p.y = dist_y(random_gen);
    p.theta = dist_theta(random_gen);
    p.weight = 1;

    particles.push_back(p);
    weights.push_back(1);
  }
  //set up init finished
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	//set up noise Gaussian
  normal_distribution<double> dist_x(0,std_pos[0]);
  normal_distribution<double> dist_y(0,std_pos[1]);
  normal_distribution<double> dist_theta(0,std_pos[2]);

  //predict each state
  for (unsigned int i=0; i < particles.size(); i++) {

    double theta = particles[i].theta;

    // yaw_raw is zero
    if (fabs(yaw_rate) < 0.001) {
      particles[i].x += velocity * delta_t * cos(theta);
      particles[i].y += velocity * delta_t * sin(theta);
    } else {
      particles[i].x += (velocity/yaw_rate) * (sin(theta + delta_t * yaw_rate) - sin(theta));
      particles[i].y += (velocity/yaw_rate) * (cos(theta) - cos(theta + delta_t * yaw_rate));
      particles[i].theta += delta_t * yaw_rate;
    }

    // add noise
    particles[i].x += dist_x(random_gen);
    particles[i].y += dist_y(random_gen);
    particles[i].theta += dist_theta(random_gen);

  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  for (unsigned int i=0; i < observations.size(); i++) {

    // init the first predicted landmark
    double closest_dist = dist(predicted[0].x,predicted[0].y,observations[i].x,observations[i].y);
    unsigned int closest_id = predicted[0].id;

    // run loop from second predicted landmark to search closest observation
    for (unsigned int j=1; j < predicted.size(); j++) {

      double separation = dist(predicted[j].x,predicted[j].y,observations[i].x,observations[i].y);

      if (separation < closest_dist) {
        closest_dist = separation;
        closest_id = predicted[j].id;
      }
    }
    observations[i].id = closest_id;
  }


}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  // step all particles
  for (unsigned int i=0; i < particles.size(); i++) {

    // filter landmark coordinates within sensor range
    std::vector<LandmarkObs> predicted;
    for (unsigned int j=0; j < map_landmarks.landmark_list.size(); j++) {

      LandmarkObs m;
      m.x = map_landmarks.landmark_list[j].x_f;
      m.y = map_landmarks.landmark_list[j].y_f;
      m.id = map_landmarks.landmark_list[j].id_i;

      // check landmark within sensor range
      if (dist(m.x,m.y,particles[i].x,particles[i].y) <= sensor_range) {
        predicted.push_back(m);
      }
    }

    // set weight to zero and go to next particle
    if (predicted.size() == 0) {
      particles[i].weight = 0;
      break;
    }

    // transform observations to map frame
    std::vector<LandmarkObs> transformed_observations;;
    for (unsigned int j=0; j < observations.size(); j++) {

      double o_x = observations[j].x;
      double o_y = observations[j].y;

      double theta = particles[i].theta;

      double p_x = cos(theta) * o_x - sin(theta) * o_y + particles[i].x;
      double p_y = sin(theta) * o_x + cos(theta) * o_y + particles[i].y;

      LandmarkObs obs;
      obs.x = p_x;
      obs.y = p_y;
      obs.id = -1;
      transformed_observations.push_back(obs);
    }

    // get data association b/t predicted landmark coordinates and observations
    dataAssociation(predicted, transformed_observations);

    // first initialize weight
    particles[i].weight = 1;

    for (unsigned int j=0; j < transformed_observations.size(); j++) {

      // find predicted landmark
      for (unsigned int k=0; k < predicted.size(); k++) {

        if (predicted[k].id == transformed_observations[j].id) {

          LandmarkObs p = predicted[k];
          LandmarkObs o = transformed_observations[j];

          // calculate normalization term
          double gauss_norm = 1.0 / (2 * M_PI * std_landmark[0] * std_landmark[1]);

          // calculate exponent
          double exponent = ( pow(p.x - o.x,2)/(2 * std_landmark[0] * std_landmark[0]) +
                  pow(p.y - o.y,2)/(2 * std_landmark[1] * std_landmark[1]) );
          // calculate weight using normalization terms and exponent
          double weight = gauss_norm * exp(-exponent);

          // calculate final weight
          particles[i].weight *= weight;
          break;
        }
      }
    }
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // get all weights
  for (unsigned int i=0; i < particles.size(); i++) {
    weights[i] = particles[i].weight;
  }

  // set up distribution
  std::discrete_distribution<> d(weights.begin(),weights.end());

  // resample
  std::vector<Particle> resampled_particles;
  for (unsigned int i=0; i < particles.size(); i++) {
    resampled_particles.push_back(particles[d(random_gen)]);
  }
  particles = std::move(resampled_particles);

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
